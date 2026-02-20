# style_judge.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from .user_simulator import STYLE_PERSONAS


class StyleJudge:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer,
        device: torch.device,
        style: str,
        max_input_tokens: int = 2048,
        tie_margin: float = 0.05,
    ):
        if style not in STYLE_PERSONAS:
            raise ValueError(f"Unknown style '{style}'. Known styles: {list(STYLE_PERSONAS.keys())}")

        self.model = model.eval()
        self.tok = tokenizer
        self.device = device
        self.style = style
        self.system_persona = STYLE_PERSONAS[style]
        self.max_input_tokens = max_input_tokens
        self.tie_margin = tie_margin

        if self.tok.pad_token is None:
            self.tok.add_special_tokens({"pad_token": "[PAD]"})
            # If you added tokens after model init, you MAY need:
            # self.model.resize_token_embeddings(len(self.tok))

    def get_system_persona(self) -> str:
        return self.system_persona

    def _build_prompt_text(self, raw_prompt: str, ca: str, cb: str) -> str:
        user_msg = (
            "You are shown two candidate responses to the same user prompt.\n\n"
            f"User prompt:\n{raw_prompt}\n\n"
            f"Response A:\n{ca}\n\n"
            f"Response B:\n{cb}\n\n"
            "As a user with the strong user persona described in the system message, "
            "decide which response you prefer.\n\n"
            "Respond with exactly one character:\n"
            "A  if you prefer Response A\n"
            "B  if you prefer Response B\n"
            "C  only if you have no preference.\n"
        )
        # user_msg = (
        #     "You are shown two candidate TL;DR summaries of the same text.\n\n"
        #     f"Original text:\n{raw_prompt}\n\n"
        #     f"Summary A:\n{ca}\n\n"
        #     f"Summary B:\n{cb}\n\n"
        #     "As a user with the strong style preference described in the system message, "
        #     "decide which summary you prefer based on style.\n\n"
        #     "Respond with exactly one character:\n"
        #     "A  if you prefer Summary A\n"
        #     "B  if you prefer Summary B\n"
        #     "C  only if you have no preference.\n"
        # )

        chat = [
            {"role": "system", "content": self.system_persona},
            {"role": "user", "content": user_msg},
        ]

        if getattr(self.tok, "apply_chat_template", None) is not None:
            return self.tok.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen-style
            )

        return f"{self.system_persona}\n\nUser:\n{user_msg}\n\nAssistant:"

    def _label_variants(self) -> Dict[str, List[List[int]]]:
        variants = {
            "A": ["A", " A", "\nA", "\n A"],
            "B": ["B", " B", "\nB", "\n B"],
            "C": ["C", " C", "\nC", "\n C"],
        }
        out: Dict[str, List[List[int]]] = {}
        for k, vs in variants.items():
            toks = []
            for s in vs:
                ids = self.tok.encode(s, add_special_tokens=False)
                if ids:
                    toks.append(ids)
            if not toks:
                raise RuntimeError(f"Tokenizer produced no ids for label {k}.")
            out[k] = toks
        return out

    @torch.no_grad()
    def _score_label_variant_batch(
        self,
        prompt_ids: torch.Tensor,          # (B, T)
        prompt_attn: torch.Tensor,         # (B, T)
        prompt_lens: torch.Tensor,         # (B,)
        label_ids: List[int],              # (L,)
    ) -> torch.Tensor:
        B = prompt_ids.size(0)
        label = torch.tensor(label_ids, dtype=torch.long, device=prompt_ids.device)  # (L,)
        L = label.numel()

        seqs = []
        attns = []
        for i in range(B):
            l_i = int(prompt_lens[i].item())
            seq = torch.cat([prompt_ids[i, :l_i], label], dim=0)
            seqs.append(seq)
            attns.append(torch.ones_like(seq, dtype=prompt_attn.dtype, device=prompt_attn.device))

        max_len = max(s.numel() for s in seqs)
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=prompt_ids.device)
        attention_mask = torch.zeros((B, max_len), dtype=prompt_attn.dtype, device=prompt_attn.device)

        for i in range(B):
            s = seqs[i]
            input_ids[i, : s.numel()] = s
            attention_mask[i, : s.numel()] = attns[i]

        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits  # (B, max_len, V)


        ar = torch.arange(L, device=prompt_ids.device).view(1, L).expand(B, L)
        pos = (prompt_lens.view(B, 1) - 1) + ar  # (B, L)

        batch_idx = torch.arange(B, device=prompt_ids.device).view(B, 1).expand(B, L)
        logits_at = logits[batch_idx, pos, :]  # (B, L, V)

        logprobs_at = F.log_softmax(logits_at.float(), dim=-1)  # (B, L, V)
        label_rep = label.view(1, L).expand(B, L)               # (B, L)
        lp = logprobs_at.gather(-1, label_rep.unsqueeze(-1)).squeeze(-1)  # (B, L)
        return lp.sum(dim=1)  # (B,)

    @torch.no_grad()
    def _score_abc_batch(self, inputs_text: List[str]) -> Dict[str, torch.Tensor]:
        """
        Returns dict with scores["A"/"B"/"C"] each shape (B,)
        """
        enc = self.tok(
            inputs_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        prompt_ids = enc["input_ids"]
        prompt_attn = enc["attention_mask"]
        prompt_lens = prompt_attn.sum(dim=1)  # (B,)

        label_variants = self._label_variants()

        scores: Dict[str, torch.Tensor] = {}
        for label, variants in label_variants.items():
            best: Optional[torch.Tensor] = None
            for v in variants:
                s = self._score_label_variant_batch(prompt_ids, prompt_attn, prompt_lens, v)  # (B,)
                best = s if best is None else torch.maximum(best, s)
            scores[label] = best
        return scores

    def _decide_from_scores(self, scores_abc: Dict[str, torch.Tensor]) -> List[int]:
        A = scores_abc["A"]
        B = scores_abc["B"]
        C = scores_abc["C"]

        # stack for argmax/second best
        stacked = torch.stack([A, B, C], dim=1)  # (B, 3)
        top2 = torch.topk(stacked, k=2, dim=1).values  # (B,2)
        gap = top2[:, 0] - top2[:, 1]  # (B,)

        best_idx = torch.argmax(stacked, dim=1)  # 0=A,1=B,2=C

        # Conservative tie: small margin or C wins
        decisions = torch.full((stacked.size(0),), -1, dtype=torch.long, device=stacked.device)

        confident = gap >= self.tie_margin
        non_c = best_idx != 2
        ab_clear = (torch.abs(A - B) >= self.tie_margin)

        choose = confident & non_c & ab_clear
        decisions[choose & (A > B)] = 0
        decisions[choose & (B > A)] = 1

        return decisions.tolist()

    @staticmethod
    def _invert_ab(decisions: List[int]) -> List[int]:
        out = []
        for d in decisions:
            if d == 0:
                out.append(1)
            elif d == 1:
                out.append(0)
            else:
                out.append(-1)
        return out

    @torch.no_grad()
    def choose_batch(
        self,
        prompts: List[str],
        completions_a: List[str],
        completions_b: List[str],
        batch_size: int = 16,
    ) -> List[int]:
        """
        Symmetric judge: runs (A,B) and (B,A), returns A/B if consistent else tie.
        """
        assert len(prompts) == len(completions_a) == len(completions_b)
        n = len(prompts)

        decisions_final: List[int] = []
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)

            # AB
            texts_ab = [
                self._build_prompt_text(prompts[i], completions_a[i], completions_b[i])
                for i in range(start, end)
            ]
            scores_ab = self._score_abc_batch(texts_ab)
            d_ab = self._decide_from_scores(scores_ab)

            # BA
            texts_ba = [
                self._build_prompt_text(prompts[i], completions_b[i], completions_a[i])
                for i in range(start, end)
            ]
            scores_ba = self._score_abc_batch(texts_ba)
            d_ba = self._decide_from_scores(scores_ba)
            d_ba_inv = self._invert_ab(d_ba)

            # agree => keep; else tie
            for x, y in zip(d_ab, d_ba_inv):
                decisions_final.append(x if x == y else -1)

        return decisions_final


    @torch.no_grad()
    def choose_batch_generated(
        self,
        prompts: List[str],
        completions_a: List[str],
        completions_b: List[str],
        batch_size: int = 8,
    ) -> List[int]:
        """
        AlpacaEval-style generation judging.
        Runs (A,B) and (B,A) and looks for 'A', 'B', or 'C' in the output text.
        """
        assert len(prompts) == len(completions_a) == len(completions_b)
        n = len(prompts)

        # Ensure tokenizer has pad token for batch generation
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        final_decisions: List[int] = []

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            batch_prompts = prompts[start:end]
            batch_a = completions_a[start:end]
            batch_b = completions_b[start:end]

            # 1. Run Forward (A, B) and Backward (B, A)
            d_ab = self._get_generation_decisions(batch_prompts, batch_a, batch_b)
            d_ba = self._get_generation_decisions(batch_prompts, batch_b, batch_a)

            # 2. Invert the BA results
            d_ba_inv = self._invert_ab(d_ba)

            # 3. Symmetry Check (Agreement)
            for x, y in zip(d_ab, d_ba_inv):
                # If they agree on A or B, keep it.
                # If they both say tie (-1), keep it.
                # If they disagree, force a tie (-1).
                final_decisions.append(x if x == y else -1)

        return final_decisions

    def _get_generation_decisions(self, p, ca, cb) -> List[int]:
        """Helper to generate text and parse for A/B/C."""
        texts = [self._build_prompt_text(p[i], ca[i], cb[i]) for i in range(len(p))]

        enc = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens
        ).to(self.device)

        # Greedy generation (AlpacaEval uses low/zero temp)
        outputs = self.model.generate(
            **enc,
            max_new_tokens=5,
            temperature=0,
            do_sample=False,
            pad_token_id=self.tok.pad_token_id
        )

        # Decode only the new tokens
        gen_texts = self.tok.batch_decode(outputs[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        batch_results = []
        for text in gen_texts:
            clean = text.strip().upper()
            if not clean:
                batch_results.append(-1)
                continue

            # AlpacaEval logic: take the first valid label found
            choice = -1
            for char in clean:
                if char == 'A':
                    choice = 0
                    break
                if char == 'B':
                    choice = 1
                    break
                if char == 'C':
                    choice = -1
                    break
            batch_results.append(choice)

        return batch_results
