"""
Core SDPO update logic for live, single-user chat training.

Owns the model, tokenizer, optimizer, and handles both
generation (eval mode) and per-turn gradient updates (train mode).
"""
import copy
import os
import shutil
import threading
import time
import traceback
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

from live_config import LiveSDPOConfig


class LiveSDPOUpdater:

    def __init__(self, config: LiveSDPOConfig):
        self.config = config
        self.step = 0
        self.metrics_history: List[Dict] = []

        # Async training state
        self._train_thread: Optional[threading.Thread] = None
        self._train_lock = threading.Lock()
        self._last_async_metrics: Optional[Dict] = None
        self._async_error: Optional[str] = None

        # vLLM state
        self.llm = None  # vLLM LLM instance (GPU 0)
        self._latest_lora_path: Optional[str] = None
        self._lora_request_id: int = 0

        # Force LoRA + async when using vLLM
        if config.use_vllm:
            if not config.use_lora:
                print("[LIVE SDPO] use_vllm=True forces use_lora=True", flush=True)
                config.use_lora = True
            if not config.async_training:
                print("[LIVE SDPO] use_vllm=True forces async_training=True", flush=True)
                config.async_training = True

        print("[LIVE SDPO] Loading model and tokenizer...", flush=True)
        self._load_model_and_tokenizer()
        print("[LIVE SDPO] Setting up optimizer...", flush=True)
        self._setup_optimizer()
        print("[LIVE SDPO] Setting up generation config...", flush=True)
        self._setup_generation_config()
        print("[LIVE SDPO] Updater ready.", flush=True)

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #

    def _load_model_and_tokenizer(self):
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path, use_fast=True, padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Training model: GPU 1 when vLLM, else auto
        device_map = {"": "cuda:1"} if self.config.use_vllm else "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            dtype=dtype,
            attn_implementation=self.config.attn_implementation,
            device_map=device_map,
        )

        if self.config.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()

        self.device = next(self.model.parameters()).device

        # vLLM inference engine on GPU 0
        if self.config.use_vllm:
            self._init_vllm()

    def _setup_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

    def _setup_generation_config(self):
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    # ------------------------------------------------------------------ #
    #  vLLM initialization
    # ------------------------------------------------------------------ #

    def _init_vllm(self):
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # ensure both GPUs visible

        from vllm import LLM
        print("[LIVE SDPO] Initializing vLLM on GPU 0...", flush=True)
        self.llm = LLM(
            model=self.config.model_name_or_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=self.config.lora_r,
            device="cuda:0",
        )
        print("[LIVE SDPO] vLLM ready.", flush=True)

    def _generate_vllm(self, messages: List[Dict[str, str]]) -> str:
        """Generate using vLLM engine on GPU 0."""
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
        )

        lora_request = None
        if self._latest_lora_path is not None:
            lora_request = LoRARequest(
                f"live_adapter_{self._lora_request_id}",
                self._lora_request_id,
                self._latest_lora_path,
            )

        outputs = self.llm.generate(
            prompt_text,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        return outputs[0].outputs[0].text

    # ------------------------------------------------------------------ #
    #  Generation (inference mode)
    # ------------------------------------------------------------------ #

    def _enter_inference_mode(self):
        self.model.eval()

    def _enter_training_mode(self):
        self.model.train()

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate an assistant response given the conversation so far."""
        if self.config.use_vllm:
            return self._generate_vllm(messages)

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        self._enter_inference_mode()
        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_context_length,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **enc, generation_config=self.generation_config,
            )
            new_tokens = output_ids[0, enc["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_response_streaming(self, messages: List[Dict[str, str]]):
        """Yield partial text chunks for Gradio streaming."""
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        self._enter_inference_mode()
        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_context_length,
        ).to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        gen_kwargs = {**enc, "generation_config": self.generation_config, "streamer": streamer}

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for chunk in streamer:
            yield chunk
        thread.join()

    # ------------------------------------------------------------------ #
    #  Hindsight prompt construction  (offline_trainer.py:64-70)
    # ------------------------------------------------------------------ #

    def _build_hindsight_messages(
        self,
        messages: List[Dict[str, str]],
        follow_up: str,
    ) -> List[Dict[str, str]]:
        """Append [HINDSIGHT CONTEXT] block to last user message."""
        conditional = copy.deepcopy(messages)
        block = self.config.hindsight_block_template.format(
            follow_up=follow_up.strip(),
        )
        # Walk backwards to find the last user message
        for i in range(len(conditional) - 1, -1, -1):
            if conditional[i]["role"] == "user":
                conditional[i]["content"] += block
                break
        return conditional

    # ------------------------------------------------------------------ #
    #  Token log-prob computation  (offline_trainer.py:393-462)
    # ------------------------------------------------------------------ #

    def _compute_token_logprobs(
        self,
        context_text: str,
        completion_ids: torch.Tensor,
        need_grad: bool = False,
        need_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Per-token log-probs of *completion_ids* given a rendered context string.

        Returns
        -------
        per_token_logps : (1, C')  after ignore_first_k
        token_mask      : (1, C')  1 = real, 0 = pad
        logits_y        : (1, C', V) or None  (only when need_logits)
        """
        tokenizer = self.tokenizer

        old_pad_side = tokenizer.padding_side
        old_trunc_side = tokenizer.truncation_side
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        enc = tokenizer(
            [context_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_context_length,
            add_special_tokens=False,
        ).to(self.device)

        tokenizer.padding_side = old_pad_side
        tokenizer.truncation_side = old_trunc_side

        y_ids = completion_ids.to(self.device)          # (1, C)
        pad_id = tokenizer.pad_token_id
        y_mask = (y_ids != pad_id).long()               # (1, C)

        input_ids = torch.cat([enc["input_ids"], y_ids], dim=1)
        attention_mask = torch.cat([enc["attention_mask"], y_mask], dim=1)

        ctx = torch.enable_grad() if need_grad else torch.no_grad()
        with ctx:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits                     # (1, S, V)

        # Shift: logits[t] predicts token[t+1]
        logits_shifted = logits[:, :-1, :]
        labels = input_ids[:, 1:]

        seq_len_y = y_ids.size(1)
        logits_y = logits_shifted[:, -seq_len_y:, :]   # (1, C, V)
        labels_y = labels[:, -seq_len_y:]               # (1, C)
        labels_y = labels_y.masked_fill(y_mask == 0, -100)

        B, C, V = logits_y.shape
        nll = F.cross_entropy(
            logits_y.reshape(B * C, V),
            labels_y.reshape(B * C),
            reduction="none",
            ignore_index=-100,
        ).reshape(B, C)

        per_token_logps = -nll                          # (1, C)

        # ignore_first_k
        k = self.config.ignore_first_k
        if k > 0 and seq_len_y > k:
            per_token_logps = per_token_logps[:, k:]
            y_mask = y_mask[:, k:]
            if need_logits:
                logits_y = logits_y[:, k:, :]

        return per_token_logps, y_mask, (logits_y if need_logits else None)

    # ------------------------------------------------------------------ #
    #  Training step
    # ------------------------------------------------------------------ #

    def train_step(
        self,
        messages_before_response: List[Dict[str, str]],
        assistant_response: str,
        user_follow_up: str,
    ) -> Dict[str, float]:
        """
        One SDPO gradient step.

        Parameters
        ----------
        messages_before_response
            Conversation up to (not including) the assistant response.  This is "x".
        assistant_response
            The assistant completion "y" shown to the user.
        user_follow_up
            The user's follow-up "o" (hindsight signal).
        """
        self._enter_training_mode()
        t0 = time.time()

        # ---- tokenize the completion ----
        completion_text = assistant_response.rstrip()
        if self.tokenizer.eos_token is not None:
            completion_text += self.tokenizer.eos_token
        completion_ids = self.tokenizer(
            [completion_text],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]  # (1, C)

        # ---- build x and xo prompts ----
        x_text = self.tokenizer.apply_chat_template(
            messages_before_response,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        xo_messages = self._build_hindsight_messages(
            messages_before_response, user_follow_up,
        )
        xo_text = self.tokenizer.apply_chat_template(
            xo_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # ---- compute loss ----
        if self.config.loss_mode == "simple_signal":
            loss, metrics = self._simple_signal_loss(x_text, xo_text, completion_ids)
        elif self.config.loss_mode == "full_distillation":
            loss, metrics = self._full_distillation_loss(x_text, xo_text, completion_ids)
        else:
            raise ValueError(f"Unknown loss_mode: {self.config.loss_mode}")

        # ---- backward + clip + step ----
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.step += 1
        metrics["step"] = self.step
        metrics["grad_norm"] = grad_norm.item()
        metrics["loss"] = loss.item()
        metrics["train_time_s"] = round(time.time() - t0, 2)
        self.metrics_history.append(metrics)

        # ---- sync LoRA adapter for vLLM ----
        if self.config.use_vllm:
            adapter_path = os.path.join(self.config.checkpoint_dir, "_live_adapter")
            os.makedirs(adapter_path, exist_ok=True)
            self.model.save_pretrained(adapter_path)
            self._latest_lora_path = adapter_path
            self._lora_request_id += 1

        # ---- auto-checkpoint ----
        if (
            self.config.checkpoint_every_n_steps > 0
            and self.step % self.config.checkpoint_every_n_steps == 0
        ):
            self.save_checkpoint()

        self._enter_inference_mode()
        return metrics

    # ------------------------------------------------------------------ #
    #  Async training
    # ------------------------------------------------------------------ #

    def wait_for_training(self) -> Optional[Dict]:
        """Block until any in-flight async training finishes. Returns metrics or None."""
        if self._train_thread is not None:
            self._train_thread.join()
            self._train_thread = None
        metrics = self._last_async_metrics
        self._last_async_metrics = None
        error = self._async_error
        self._async_error = None
        if error:
            print(f"[LIVE SDPO] Async training error: {error}", flush=True)
        return metrics

    def train_step_async(
        self,
        messages_before_response: List[Dict[str, str]],
        assistant_response: str,
        user_follow_up: str,
    ) -> None:
        """Launch train_step in a background thread."""
        # Ensure any previous async step is done before starting a new one
        self.wait_for_training()

        def _run():
            try:
                with self._train_lock:
                    metrics = self.train_step(
                        messages_before_response=messages_before_response,
                        assistant_response=assistant_response,
                        user_follow_up=user_follow_up,
                    )
                self._last_async_metrics = metrics
            except Exception as e:
                self._async_error = f"{e}\n{traceback.format_exc()}"

        self._train_thread = threading.Thread(target=_run, daemon=True)
        self._train_thread.start()

    # ------------------------------------------------------------------ #
    #  Loss functions
    # ------------------------------------------------------------------ #

    def _simple_signal_loss(
        self,
        x_text: str,
        xo_text: str,
        completion_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        loss = -(signal.detach() * log pi(y|x)),  length-normalised.
        Ported from offline_trainer.py:256-294.
        """
        # log pi(y | x, o) — detached, no grad
        with torch.no_grad():
            logps_xo, mask_xo, _ = self._compute_token_logprobs(
                xo_text, completion_ids, need_grad=False,
            )

        # log pi(y | x) — with grad
        logps_x, mask_x, _ = self._compute_token_logprobs(
            x_text, completion_ids, need_grad=True,
        )

        mask_f = mask_x.float()
        length = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)

        per_token_diff = (logps_xo - logps_x).detach()
        if self.config.signal_clip > 0:
            per_token_diff = per_token_diff.clamp(
                -self.config.signal_clip, self.config.signal_clip,
            )
        per_token_loss = -(per_token_diff * logps_x) * mask_f
        loss = (per_token_loss.sum(dim=1, keepdim=True) / length).mean()

        total_tokens = mask_f.sum().clamp(min=1.0)
        metrics = {
            "signal_mean": (per_token_diff * mask_f).sum().item() / total_tokens.item(),
            "signal_std": (
                per_token_diff[mask_x.bool()].std().item()
                if mask_x.any() else 0.0
            ),
            "policy_logp_mean": (logps_x.detach() * mask_f).sum().item() / total_tokens.item(),
            "critic_logp_mean": (logps_xo * mask_f).sum().item() / total_tokens.item(),
            "completion_tokens": int(total_tokens.item()),
        }

        self._log_token_table(completion_ids, mask_x, logps_x, logps_xo, per_token_diff)

        return loss, metrics

    def _full_distillation_loss(
        self,
        x_text: str,
        xo_text: str,
        completion_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Reverse KL top-k distillation: KL(student || teacher).

        Student = pi(·|x), Teacher = pi(·|x,o).
        Top-k indices are taken from the student (on-policy), and the
        teacher is evaluated on those same indices.
        """
        # Teacher: pi(·|x,o) — no grad
        with torch.no_grad():
            _, _, logits_xo = self._compute_token_logprobs(
                xo_text, completion_ids, need_grad=False, need_logits=True,
            )

        # Student: pi(·|x) — with grad
        _, mask_x, logits_x = self._compute_token_logprobs(
            x_text, completion_ids, need_grad=True, need_logits=True,
        )

        mask_f = mask_x.float()
        total_tokens = mask_f.sum().clamp(min=1.0)
        K = self.config.distillation_topk

        # Full log-softmax over vocab
        student_log_probs = F.log_softmax(logits_x, dim=-1)   # (1, C', V)
        with torch.no_grad():
            teacher_log_probs = F.log_softmax(logits_xo, dim=-1)  # (1, C', V)

        # Student's top-k indices (on-policy support)
        student_topk, topk_idx = torch.topk(student_log_probs, k=K, dim=-1)  # (1, C', K)
        teacher_topk = torch.gather(teacher_log_probs, dim=-1, index=topk_idx)  # (1, C', K)

        # Tail handling
        if self.config.distillation_add_tail:
            student_topk = self._add_tail(student_topk)   # (1, C', K+1)
            teacher_topk = self._add_tail(teacher_topk)   # (1, C', K+1)
        else:
            student_topk = self._renorm_topk(student_topk)
            teacher_topk = self._renorm_topk(teacher_topk)

        # Reverse KL: KL(student || teacher) = sum_x student(x) * (log student(x) - log teacher(x))
        # PyTorch kl_div(input, target) = target * (log target - input) when log_target=True
        # So KL(student || teacher) = kl_div(input=teacher, target=student, log_target=True)
        kl_per_token_per_k = F.kl_div(
            teacher_topk, student_topk, reduction="none", log_target=True,
        )  # (1, C', K+1) or (1, C', K)
        per_token_kl = kl_per_token_per_k.sum(dim=-1)  # (1, C')

        # Masked mean over tokens
        loss = (per_token_kl * mask_f).sum() / total_tokens

        metrics = {
            "kl_mean": (per_token_kl.detach() * mask_f).sum().item() / total_tokens.item(),
            "completion_tokens": int(total_tokens.item()),
            "loss_mode": "full_distillation",
        }
        return loss, metrics

    @staticmethod
    def _add_tail(log_probs: torch.Tensor) -> torch.Tensor:
        """Append log-prob of the tail (1 - sum(top_k_probs)) as a K+1th entry."""
        log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([log_probs, tail_log], dim=-1)

    @staticmethod
    def _renorm_topk(log_probs: torch.Tensor) -> torch.Tensor:
        """Renormalize top-k log-probs to form a valid distribution."""
        logZ = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        return log_probs - logZ

    # ------------------------------------------------------------------ #
    #  Debug logging  (offline_trainer.py:466-536)
    # ------------------------------------------------------------------ #

    def _log_token_table(
        self,
        completion_ids: torch.Tensor,
        mask: torch.Tensor,
        logps_x: torch.Tensor,
        logps_xo: torch.Tensor,
        signal: torch.Tensor,
    ):
        MAX_TOKENS = 200

        mask_bool = mask[0].detach().cpu().bool()
        if not mask_bool.any():
            return

        # Pad completion_ids to match mask length (after ignore_first_k)
        k = self.config.ignore_first_k
        c_ids = completion_ids[0].detach().cpu()
        if k > 0 and c_ids.size(0) > k:
            c_ids = c_ids[k:]

        tok_ids = c_ids[mask_bool].tolist()
        lp_x = logps_x[0].detach().cpu()[mask_bool].tolist()
        lp_xo = logps_xo[0].detach().cpu()[mask_bool].tolist()
        ratios = signal[0].detach().cpu()[mask_bool].tolist()

        n = min(len(tok_ids), MAX_TOKENS)

        tok_strings = [
            self.tokenizer.decode([tid], clean_up_tokenization_spaces=False).replace("\n", "\\n")
            for tid in tok_ids[:n]
        ]

        print(f"\n[LIVE SDPO] Step {self.step}  token-wise log-probs:")
        header = f"{'idx':>4} | {'tok_id':>7} | {'tok_str':<15} | {'logp_x':>12} | {'logp_xo':>12} | {'signal':>12}"
        print(header)
        print("-" * len(header))

        for i in range(n):
            tstr = (tok_strings[i][:12] + "…") if len(tok_strings[i]) > 15 else tok_strings[i]
            print(
                f"{i:4d} | {tok_ids[i]:7d} | {tstr:<15} | "
                f"{lp_x[i]:12.6f} | {lp_xo[i]:12.6f} | {ratios[i]:12.6f}"
            )

        if len(tok_ids) > MAX_TOKENS:
            print(f"... (truncated to first {MAX_TOKENS} tokens)")
        print()

    # ------------------------------------------------------------------ #
    #  Checkpointing
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        tag = tag or f"step_{self.step}"
        ckpt_path = os.path.join(self.config.checkpoint_dir, tag)
        os.makedirs(ckpt_path, exist_ok=True)

        self.model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)

        torch.save(
            {"optimizer_state_dict": self.optimizer.state_dict(), "step": self.step},
            os.path.join(ckpt_path, "training_state.pt"),
        )

        self._cleanup_old_checkpoints()
        print(f"[LIVE SDPO] Checkpoint saved → {ckpt_path}")
        return ckpt_path

    def _cleanup_old_checkpoints(self):
        if not os.path.exists(self.config.checkpoint_dir):
            return
        ckpts = sorted(
            [
                d for d in os.listdir(self.config.checkpoint_dir)
                if os.path.isdir(os.path.join(self.config.checkpoint_dir, d))
            ],
            key=lambda d: os.path.getmtime(
                os.path.join(self.config.checkpoint_dir, d)
            ),
        )
        while len(ckpts) > self.config.max_checkpoints:
            oldest = ckpts.pop(0)
            shutil.rmtree(os.path.join(self.config.checkpoint_dir, oldest))
