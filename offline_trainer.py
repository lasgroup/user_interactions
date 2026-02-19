# lras_offline_trainer.py
import torch
import copy
from dataclasses import dataclass
from typing import Any, List, Dict
from transformers import PreTrainedTokenizerBase, GenerationConfig
import torch.nn.functional as F
from transformers import Trainer
from collections import defaultdict
from typing import Any, List, Dict, Optional, Union


@dataclass
class OfflineLRASCollator:
    tokenizer: PreTrainedTokenizerBase
    max_completion_length: int = 2048
    
    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Converts 'from/value' (WildChat) format to 'role/content' (Standard) format.
        """
        normalized = []
        for msg in messages:
            # Check if we need to convert keys
            if "value" in msg and "content" not in msg:
                # Map roles: human->user, gpt->assistant
                role_map = {"human": "user", "gpt": "assistant", "system": "system"}
                original_role = msg.get("from", "user")
                new_role = role_map.get(original_role, original_role)
                
                normalized.append({
                    "role": new_role,
                    "content": msg["value"]
                })
            else:
                # Already in correct format
                normalized.append(msg)
        return normalized

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_texts = []
        conditional_texts = []
        completion_texts = []
        
        for ex in examples:
            clean_prompt = self._normalize_messages(ex["prompt"])
            
            # --- Standard Prompt (x) ---
            p_text = self.tokenizer.apply_chat_template(
                clean_prompt, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_texts.append(p_text)

            
            # --- Conditional Prompt (x, o) ---
            conditional_history = clean_prompt[:]

            fb = ex["user_response"].get("value") or ex["user_response"].get("content")
            o = fb.strip()
            
            conditional_history = copy.deepcopy(clean_prompt)
            block = (
                "\n\n[HINDSIGHT CONTEXT]\n"
                "The following is a user response to your previous, insufficient attempt. Improve your response to the user prompt.\n" # Do not respond to the future user message.\n"
                f"Future User Message: {o}"
            )
            conditional_history[-1]["content"] += block

            # previous version used for Qwen3-4B debugging
            # conditional_history.append({
            #     "role": "assistant",
            #     "content": (
            #     "=== HINDSIGHT CONTEXT ===\n"
            #     "[The following is a future user message. Use this to guide your answer to the user prompt.]\n"
            #     f"{o}"
            #     )
            # })


            xo_text = self.tokenizer.apply_chat_template(
                conditional_history,
                tokenize=False,
                add_generation_prompt=True, 
                enable_thinking=False,
            )
            conditional_texts.append(xo_text)

            text = ex["completion"].get("value") or ex["completion"].get("content")
            text = text.rstrip()

            # Explicitly append EOS so EOS gets LRAS gradients
            if self.tokenizer.eos_token is not None:
                text = text + self.tokenizer.eos_token

            completion_texts.append(text)


        # Tokenize
        completion_tokenized = self.tokenizer(
            completion_texts,
            padding=True,
            truncation=True,
            max_length=self.max_completion_length,
            add_special_tokens=False, 
            return_tensors="pt"
        )
        
        return {
            "prompt_texts": prompt_texts,
            "conditional_texts": conditional_texts,
            "completion_ids": completion_tokenized["input_ids"],
            "completion_mask": completion_tokenized["attention_mask"],
        }



class OfflineLRASTrainer(Trainer):
    def __init__(
        self, 
        signal_clip: float = 2,
        ignore_first_k: int = 0,
        ref_model: Optional[torch.nn.Module] = None,
        kl_beta: float = 0.0,
        kl_max_new_tokens: int = 256,
        kl_do_sample: bool = True,
        kl_temperature: float = 1.0,
        kl_top_p: float = 1.0,
        kl_once_per_optimizer_step: bool = True,
        *args, 
        **kwargs
    ):
        self.signal_clip = signal_clip
        self.ignore_first_k = ignore_first_k
        
        self._metrics_buffer = defaultdict(list)

        self._example_counter = 0 

        # KL config: not used
        self.ref_model = ref_model
        self.kl_beta = kl_beta
        self.kl_max_new_tokens = kl_max_new_tokens
        self.kl_do_sample = kl_do_sample
        self.kl_temperature = kl_temperature
        self.kl_top_p = kl_top_p
        self.kl_once_per_optimizer_step = kl_once_per_optimizer_step
        
        super().__init__(*args, **kwargs)

        self._micro_in_step = 0

    @torch.no_grad()
    def _rollout_from_policy(self, model, prompt_texts):
        tok = self.processing_class
        device = next(model.parameters()).device  # robust under wrappers

        enc = tok(
            text=prompt_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            add_special_tokens=False,
        ).to(device)

        input_len = enc["input_ids"].shape[1]
        eos_id = tok.eos_token_id
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

        gen_cfg = GenerationConfig(
            max_new_tokens=self.kl_max_new_tokens,
            do_sample=self.kl_do_sample,
            temperature=self.kl_temperature,
            top_p=self.kl_top_p,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            return_dict_in_generate=False,
        )

        gen_model = model.module if hasattr(model, "module") else model

        was_training = gen_model.training
        gen_model.eval()
        seq = gen_model.generate(**enc, generation_config=gen_cfg)
        if was_training:
            gen_model.train()

        completion_ids = seq[:, input_len:]
        return completion_ids


    def _rollout_kl_penalty(self, model, prompt_texts: List[str]) -> torch.Tensor:
        """
        Computes mean per-sequence KL(pi_theta || pi_ref) on on-policy rollouts.
        Returns scalar tensor (requires grad).
        """
        if self.ref_model is None or self.kl_beta == 0.0:
            return torch.zeros([], device=model.device, dtype=torch.float32)

        # ensure ref model on same device (do this once; cheap check)
        if next(self.ref_model.parameters()).device != model.device:
            self.ref_model.to(model.device)

        # rollout y ~ pi_theta(.|x) (no grad)
        completion_ids = self._rollout_from_policy(model, prompt_texts)  # (B, Tgen)

        # log pi_theta(y|x) with grad
        logps_pol, _, y_mask = self._token_logps_of_given_y(
            context_texts=prompt_texts,
            completion_ids_list=completion_ids,
            model=model,
        )

        # log rollout preview + length (uses y_mask)
        with torch.no_grad():
            self._maybe_log_kl_rollout_preview(
                prompt_texts=prompt_texts,
                completion_ids=completion_ids,
                y_mask=y_mask,
                max_preview_tokens=150,
                every_n_steps=5,
            )
        
        # log pi_ref(y|x) no grad
        with torch.no_grad():
            logps_ref, _, _ = self._token_logps_of_given_y(
                context_texts=prompt_texts,
                completion_ids_list=completion_ids,
                model=self.ref_model,
            )

        y_mask_f = y_mask.float()
        lengths = y_mask_f.sum(dim=1).clamp(min=1.0)

        # delta is the per-token advantage 
        delta = (logps_pol - logps_ref).detach()

        # reverse-KL surrogate
        kl_sur = ((delta + 1.0) * logps_pol * y_mask_f).sum(dim=1) / lengths
        kl_tok_mean = kl_sur.mean()

        # logging
        with torch.no_grad():
            per_token_kl_val = (logps_pol.detach() - logps_ref) * y_mask_f
            kl_seq_val = per_token_kl_val.sum(dim=1)
            kl_tok_val = kl_seq_val / lengths
            self._metrics_buffer["kl/rollout_tok_mean"].append(float(kl_tok_val.mean()))
            self._metrics_buffer["kl/rollout_seq_mean"].append(float(kl_seq_val.mean()))
            self._metrics_buffer["kl/rollout_len_mean"].append(float(lengths.mean()))

        return kl_tok_mean
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Offline loss with per-token updates and length-normalized (per-sequence
        mean-centered) signal. 
        """
        x_texts = inputs["prompt_texts"]
        xo_texts = inputs["conditional_texts"]
        completion_ids = inputs["completion_ids"]  # (B, C)

        # log pi(y | x, o)
        with torch.no_grad():
            logps_xo, _, _ = self._token_logps_of_given_y(
                    context_texts=xo_texts,
                    completion_ids_list=completion_ids,
                    model=model,
                )  # (B, C')

        # log pi(y | x)
        logps_x, y_ids, token_mask = self._token_logps_of_given_y(
            context_texts=x_texts,
            completion_ids_list=completion_ids,
            model=model,
        )  # logps_x: (B, C'), token_mask: (B, C')

        token_mask_f = token_mask.float()                           # (B, C')
        token_lengths = token_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
        total_active_tokens = token_mask_f.sum() 


        c = float(self.signal_clip)

        per_token_diff = (logps_xo - logps_x).detach() 

        per_token_diff = per_token_diff.clamp(min=-c, max=c)

        per_token_loss = -(per_token_diff * logps_x) * token_mask_f          # (B, C')

        loss_per_seq = per_token_loss.sum(dim=1, keepdim=True) / token_lengths
        loss = loss_per_seq.mean()
        base_loss = loss

        kl_mean = None
        do_kl = self.kl_beta != 0.0
        if do_kl and self.kl_once_per_optimizer_step:
            do_kl = (self._micro_in_step == 0)

        if do_kl:
            kl_mean = self._rollout_kl_penalty(model, x_texts)  # scalar
            loss = base_loss + (self.kl_beta * kl_mean)
        else:
            loss = base_loss

        # update microbatch counter
        if model.training and self.kl_once_per_optimizer_step:
            ga = int(getattr(self.args, "gradient_accumulation_steps", 1))
            self._micro_in_step = (self._micro_in_step + 1) % max(ga, 1)

        # Printing of log ratios 
        if model.training and self.is_world_process_zero():
            batch_size = y_ids.size(0)
            start_idx = self._example_counter
            global_indices = [start_idx + i for i in range(batch_size)]
            self._example_counter = start_idx + batch_size

            with torch.no_grad():
                self._maybe_log_token_table(
                    global_indices=global_indices,
                    y_ids=y_ids,
                    y_mask=token_mask,
                    logps_x=logps_x,
                    logps_xo=logps_xo,
                    per_tok_signal=per_token_diff,
                    xo_texts=xo_texts,
                )

        # Diagnostics for EOS token
        tokenizer = self.processing_class
        eos_id = tokenizer.eos_token_id
        eos_mask = (y_ids == eos_id) & token_mask.bool() 

        if model.training:
            with torch.no_grad():
                self._metrics_buffer["lras/signal_mean"].append(
                    (per_token_diff * token_mask_f).sum().item() / total_active_tokens.item()
                )
                seq_logratio = (per_token_diff * token_mask_f).sum(dim=1)        # (B,)
                length_norm_signal = seq_logratio / token_lengths.squeeze(1)     # (B,)
                self._metrics_buffer["lras/len_signal_mean"].append(
                    length_norm_signal.mean().item()
                )
                self._metrics_buffer["lras/base_loss"].append(base_loss.detach().float().item())

                self._metrics_buffer["lras/signal_std"].append(
                    per_token_diff[token_mask.bool()].std().item()
                )
                self._metrics_buffer["lras/policy_logp"].append(
                    (logps_x * token_mask_f).sum().item() / total_active_tokens.item()
                )
                self._metrics_buffer["lras/critic_logp"].append(
                    (logps_xo * token_mask_f).sum().item() / total_active_tokens.item()
                )
                if kl_mean is not None:
                    self._metrics_buffer["lras/kl_mean"].append(kl_mean.detach().float().item())
                    self._metrics_buffer["lras/kl_term"].append((self.kl_beta * kl_mean).detach().float().item())
                    
                # self._metrics_buffer["lras/loss"].append(loss.item())
                if eos_mask.any():
                    eos_signal = per_token_diff[eos_mask]
                    eos_logp_x = logps_x[eos_mask]
                    eos_logp_xo = logps_xo[eos_mask]

                    self._metrics_buffer["lras/eos_signal_mean"].append(
                        eos_signal.mean().item()
                    )
                    self._metrics_buffer["lras/eos_logp_mean"].append(
                        eos_logp_x.mean().item()
                    )
                    self._metrics_buffer["lras/eos_logratio_mean"].append(
                        (eos_logp_xo - eos_logp_x).mean().item()
                    )

        return (loss, None) if return_outputs else loss


    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Intercepts the Trainer's log call to inject our buffered metrics.
        """
        if self._metrics_buffer:
            for key, values in self._metrics_buffer.items():
                if len(values) > 0:
                    logs[key] = sum(values) / len(values)
            self._metrics_buffer.clear()
            
        super().log(logs, start_time)


    def _token_logps_of_given_y(self, context_texts, completion_ids_list, model):
        """
        Calculates log probs of y given x.
        Handles the complexity of left-padding x and right-padding y.

        Returns:
            completion_logprobs: (B, C')  log p(y_t | prefix) for completion tokens
            y_ids:               (B, C')  completion token ids after ignore_first_k
            y_mask:              (B, C')  1 for real tokens, 0 for padding, after ignore_first_k
        """
        tokenizer = self.processing_class
        device = model.device

        old_pad_side = tokenizer.padding_side
        old_trunc_side = tokenizer.truncation_side
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left" 

        enc = tokenizer(
            context_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        ).to(device)

        tokenizer.padding_side = old_pad_side
        tokenizer.truncation_side = old_trunc_side

        # Prepare completion (right pad)
        y_ids = completion_ids_list.to(device)          # (B, C)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("pad_token_id must be set and distinct from eos_token_id")
        y_mask = (y_ids != pad_id).long()               # (B, C)

        # Concatenate [context, completion]
        input_ids = torch.cat([enc["input_ids"], y_ids], dim=1)
        attention_mask = torch.cat([enc["attention_mask"], y_mask], dim=1)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, S, V)

        logits = logits[:, :-1, :]      # (B, S-1, V)
        labels = input_ids[:, 1:]       # (B, S-1)

        seq_len_y = y_ids.size(1)       # (B, C)
        logits_y = logits[:, -seq_len_y:, :]   # (B, C, V)
        labels_y = labels[:, -seq_len_y:]      # (B, C)

        labels_y = labels_y.masked_fill(y_mask == 0, -100)

        B, C, V = logits_y.shape
        nll = F.cross_entropy(
            logits_y.reshape(B * C, V),
            labels_y.reshape(B * C),
            reduction="none",
            ignore_index=-100,
        ).reshape(B, C)

        completion_logprobs = -nll  # (B, C)

        # Apply ignore_first_k consistently to both logprobs and mask
        if self.ignore_first_k > 0 and seq_len_y > self.ignore_first_k:
            completion_logprobs = completion_logprobs[:, self.ignore_first_k:]
            y_ids = y_ids[:, self.ignore_first_k:]
            y_mask = y_mask[:, self.ignore_first_k:]

        return completion_logprobs, y_ids, y_mask



    def _maybe_log_token_table(
        self,
        global_indices,      # List[int], global example indices for this batch
        y_ids,               # (B, C') tensor, after ignore_first_k
        y_mask,              # (B, C') tensor, after ignore_first_k
        logps_x,             # (B, C') tensor
        logps_xo,            # (B, C') tensor
        per_tok_signal,      # (B, C') tensor (logps_xo - logps_x)
        xo_texts,
    ):
        """
        For any example whose global index % 100 == 0, print a table:
        token | token_str | logp_x | logp_xo | log_ratio
        """
        tokenizer = self.processing_class
        MAX_TOKENS_TO_PRINT = 200

        # Move to CPU / detach once
        y_ids_cpu = y_ids.detach().cpu()
        y_mask_cpu = y_mask.detach().cpu()
        logps_x_cpu = logps_x.detach().cpu()
        logps_xo_cpu = logps_xo.detach().cpu()
        signal_cpu = per_tok_signal.detach().cpu()

        for b_idx, g_idx in enumerate(global_indices):
            if g_idx % 200 != 0:
                continue

            mask = y_mask_cpu[b_idx].bool()
            if mask.sum() == 0:
                continue

            tok_ids = y_ids_cpu[b_idx][mask].tolist()
            lp_x = logps_x_cpu[b_idx][mask].tolist()
            lp_xo = logps_xo_cpu[b_idx][mask].tolist()
            ratios = signal_cpu[b_idx][mask].tolist()

            n = min(len(tok_ids), MAX_TOKENS_TO_PRINT)
            tok_ids = tok_ids[:n]
            lp_x = lp_x[:n]
            lp_xo = lp_xo[:n]
            ratios = ratios[:n]

            tok_strings = [
                tokenizer.decode([tid], clean_up_tokenization_spaces=False).replace("\n", "\\n")
                for tid in tok_ids
            ]

            xo = xo_texts[b_idx]
            xo_disp = xo 
            MAX_CHARS = 2000
            if len(xo_disp) > MAX_CHARS:
                xo_disp = xo_disp[:MAX_CHARS] + "… [truncated]"
            print(f"[LRAS DEBUG] xo_text:\n{xo_disp}\n")

            print(f"\n[LRAS DEBUG] Example #{g_idx} (after ignore_first_k) token-wise logps:")
            header = f"{'idx':>4} | {'tok_id':>7} | {'tok_str':<15} | {'logp_x':>10} | {'logp_xo':>10} | {'log_ratio':>10}"
            print(header)
            print("-" * len(header))

            for i, (tid, tstr, lx, lxo, r) in enumerate(zip(tok_ids, tok_strings, lp_x, lp_xo, ratios)):
                tstr_display = (tstr[:12] + "…") if len(tstr) > 15 else tstr
                print(
                    f"{i:4d} | {tid:7d} | {tstr_display:<15} | "
                    f"{lx:12.8f} | {lxo:12.8f} | {r:12.8f}"
                )

            if len(y_ids_cpu[b_idx][mask]) > MAX_TOKENS_TO_PRINT:
                print(f"... (truncated to first {MAX_TOKENS_TO_PRINT} tokens)")

            print("") 


    def _maybe_log_kl_rollout_preview(
        self,
        prompt_texts: List[str],
        completion_ids: torch.Tensor,      # (B, T) right-padded ids
        y_mask: torch.Tensor,              # (B, T) 1 for real tokens
        max_preview_tokens: int = 150,
        every_n_steps: int = 5,
    ):
        if not (self.is_world_process_zero() and self.model.training):
            return

        if (self.state.global_step % every_n_steps) != 0:
            return

        tok = self.processing_class
        comp_ids_cpu = completion_ids.detach().cpu()
        mask_cpu = y_mask.detach().cpu()

        b = 0
        length = int(mask_cpu[b].sum().item())
        ids = comp_ids_cpu[b, :length].tolist()

        preview_ids = ids[:max_preview_tokens]
        preview_text = tok.decode(preview_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        preview_text = preview_text.replace("\n", "\\n")

        print(f"\n[KL ROLLOUT] step={int(self.state.global_step)}  completion_len={length}")
        print(f"[KL ROLLOUT] completion_preview (first {min(max_preview_tokens, length)} toks):")
        print(preview_text)
        print("")
