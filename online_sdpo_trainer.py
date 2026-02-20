import json
import os
from collections import deque
from contextlib import nullcontext
from typing import Any, Optional, Union

import torch
from torch import nn
from accelerate.utils import broadcast_object_list, gather, gather_object
from transformers.utils import is_flash_attn_2_available

from trl.data_utils import is_conversational, prepare_multimodal_messages
from trl.extras.profiling import profiling_context
from trl.models import unwrap_model_for_generation
from trl.trainer.base_trainer import BaseTrainer
from trl.trainer.rloo_trainer import RLOOTrainer
from trl.trainer.utils import nanmax, nanmin, pad

from user_simulator import UserSimulator, StyleUserSimulator


class SDPOOnlineTrainer(RLOOTrainer):
    def __init__(self, *args, **kwargs):
        user_model = kwargs.pop("user_model", None)
        super().__init__(*args, **kwargs)

        self._logs.setdefault("user_response", deque(maxlen=self.args.generation_batch_size))
        self._logs.setdefault("logp_y_given_x", deque(maxlen=self.args.generation_batch_size))
        self._logs.setdefault("logp_y_given_xo", deque(maxlen=self.args.generation_batch_size))

        self.debug_token_table_every = getattr(self.args, "debug_token_table_every", 1)
        self._last_debug_table_step = -1

        self.use_sdpo_signal = True
        self.ignore_first_k = 0

        self._token_logs = deque(maxlen=16)
        self.log_token_logps = True
        self.max_token_log_examples_per_step = 4

        output_dir = getattr(self.args, "output_dir", ".")
        os.makedirs(output_dir, exist_ok=True)
        self.token_log_file = os.path.join(output_dir, "token_logps.jsonl")

        style = getattr(self.args, "style", "concise_casual_beginner")

        if isinstance(user_model, UserSimulator):
            self.user_simulator = user_model
        else:
            self.user_simulator = StyleUserSimulator(
                model=user_model,
                tokenizer=self.processing_class,
                device=self.accelerator.device,
                style=style,
            )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        images = None
        prompts = [x["prompt"] for x in inputs]
        prompt_ids_list, completion_ids_list, forward_kwargs = self._generate(prompts, images)

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        raw_prompts = [ex["raw_prompt"] for ex in inputs]
        prompts_for_user = raw_prompts
        completions_for_user = completions_text

        user_responses_local = self._distributed_generate_user_feedback(
            prompts_for_user=prompts_for_user,
            completions_for_user=completions_for_user,
        )

        system_prompt = self.args.system_prompt
        x_contexts = [ex["prompt"] for ex in inputs]

        conditional_msgs = []
        for raw, uresp in zip(raw_prompts, user_responses_local):
            block = (
                "\n\n=== HINDSIGHT CONTEXT ===\n"
                "[The following is a future user message. Use this to guide your answer to the user prompt.]\n"
                f"{uresp.strip()}"
            )
            conditional_msgs.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw + block},
                ]
            )

        conditional_contexts = [
            self.processing_class.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for m in conditional_msgs
        ]

        tok_logps_x_local, tok_ids_local, tok_mask_local = self._token_logps_of_given_y(
            context_texts=x_contexts,
            completion_ids_list=completion_ids_list,
            ignore_first_k=0,
            model=self.model,
        )

        tok_logps_xo_local, _, _ = self._token_logps_of_given_y(
            context_texts=conditional_contexts,
            completion_ids_list=completion_ids_list,
            ignore_first_k=0,
            model=self.model,
        )

        if self.model.training:
            step = int(self.state.global_step)
            if (
                self.debug_token_table_every > 0
                and step % self.debug_token_table_every == 0
                and step != self._last_debug_table_step
            ):
                self._last_debug_table_step = step
                per_tok_signal = (tok_logps_xo_local - tok_logps_x_local) * tok_mask_local
                i = 0
                self._maybe_log_token_table(
                    global_indices=[step],
                    y_ids=tok_ids_local[i : i + 1],
                    y_mask=tok_mask_local[i : i + 1],
                    logps_x=tok_logps_x_local[i : i + 1],
                    logps_xo=tok_logps_xo_local[i : i + 1],
                    per_tok_signal=per_tok_signal[i : i + 1],
                    raw_prompt=raw_prompts[i],
                    completion=completions_text[i],
                    user_response=user_responses_local[i],
                    conditional_contexts=conditional_contexts[i],
                )

        logp_y_given_x_local = (tok_logps_x_local * tok_mask_local).sum(1)
        logp_y_given_xo_local = (tok_logps_xo_local * tok_mask_local).sum(1)

        per_tok_logratio_local = (tok_logps_xo_local - tok_logps_x_local) * tok_mask_local
        seq_logratio_local = per_tok_logratio_local.sum(dim=1)
        per_tok_adv_local = per_tok_logratio_local.detach()

        token_lengths_local = tok_mask_local.sum(1).float()
        token_lengths = gather(token_lengths_local).clamp(min=1.0)

        seq_logratio_global = gather(seq_logratio_local)
        sdpo_signal = seq_logratio_global / token_lengths

        if self.beta != 0.0 and self.ref_model is not None:
            tok_logps_ref_local, _, _ = self._token_logps_of_given_y(
                context_texts=x_contexts,
                completion_ids_list=completion_ids_list,
                ignore_first_k=0,
                model=self.ref_model,
            )
        else:
            tok_logps_ref_local = None

        if self.beta != 0.0 and tok_logps_ref_local is not None:
            per_token_kl_local = (tok_logps_x_local - tok_logps_ref_local) * tok_mask_local
            kl_local = per_token_kl_local.sum(dim=1) / token_lengths_local.clamp(min=1.0)
            kl = gather(kl_local)
            kl_per_seq = kl
            self._metrics[mode]["kl"].append(kl.mean().item())
        else:
            kl_per_seq = torch.zeros_like(sdpo_signal)

        sdpo_signal = sdpo_signal - self.beta * kl_per_seq

        raw_logratio = seq_logratio_global / token_lengths
        self._metrics[mode]["logratio_mean"].append(raw_logratio.mean().item())
        self._metrics[mode]["sdpo_signal_mean"].append(sdpo_signal.mean().item())

        if self.use_sdpo_signal:
            group_signal = sdpo_signal.view(-1, self.num_generations)
            if self.num_generations == 1:
                advantages = group_signal.view(-1)
            else:
                grouped_sum = group_signal.sum(dim=1, keepdim=True)
                baselines = (grouped_sum - group_signal) / (self.num_generations - 1)
                baselines = baselines.view(-1)
                advantages = sdpo_signal - baselines

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        if self.log_token_logps and self.accelerator.is_main_process:
            B, _ = tok_ids_local.shape
            num_to_log = min(self.max_token_log_examples_per_step, B)

            tok_ids_np = tok_ids_local[:num_to_log].detach().cpu()
            tok_mask_np = tok_mask_local[:num_to_log].detach().cpu()
            tok_x_np = tok_logps_x_local[:num_to_log].detach().cpu()
            tok_xo_np = tok_logps_xo_local[:num_to_log].detach().cpu()

            sampled_completions = completions_text[:num_to_log]
            sampled_prompts = prompts_text[:num_to_log]

            for i in range(num_to_log):
                mask_i = tok_mask_np[i].bool()
                ids_i = tok_ids_np[i][mask_i].tolist()
                lp_x_i = tok_x_np[i][mask_i].tolist()
                lp_xo_i = tok_xo_np[i][mask_i].tolist()

                tokens_i = [self.processing_class.decode([tid], skip_special_tokens=False) for tid in ids_i]
                log_ratio_i = [xo - x for x, xo in zip(lp_x_i, lp_xo_i)]

                self._token_logs.append(
                    {
                        "prompt": sampled_prompts[i],
                        "completion": sampled_completions[i],
                        "tokens": tokens_i,
                        "token_ids": ids_i,
                        "logp_y_given_x": lp_x_i,
                        "logp_y_given_xo": lp_xo_i,
                        "log_ratio": log_ratio_i,
                    }
                )

        self._logs["prompt"].extend(gather_object(x_contexts))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        self._logs["user_response"].extend(gather_object(user_responses_local))
        self._logs["logp_y_given_x"].extend(gather_object(logp_y_given_x_local.tolist()))
        self._logs["logp_y_given_xo"].extend(gather_object(logp_y_given_xo_local.tolist()))

        output: dict[str, Union[torch.Tensor, Any]] = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": tok_logps_x_local,
            "token_advantages": per_tok_adv_local,
            "token_mask": tok_mask_local,
        }

        if tok_logps_ref_local is not None:
            output["old_ref_per_token_logps"] = tok_logps_ref_local

        for k in [
            "pixel_values",
            "image_grid_thw",
            "pixel_attention_mask",
            "image_sizes",
            "token_type_ids",
        ]:
            if k in forward_kwargs:
                output[k] = forward_kwargs[k]

        if images is not None:
            output["num_images"] = [len(img_list) for img_list in images]

        return output

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps_full, entropies_full = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        per_token_logps = per_token_logps_full[:, self.ignore_first_k :]
        entropies = entropies_full[:, self.ignore_first_k :]

        old_per_token_logps = inputs["old_per_token_logps"]
        token_advantages = inputs["token_advantages"]
        token_mask = inputs["token_mask"]

        token_lengths = token_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_adv = (token_advantages * token_mask).sum(dim=1, keepdim=True) / token_lengths
        token_advantages = (token_advantages - mean_adv) * token_mask

        token_log_ratio = per_token_logps - old_per_token_logps
        coef_1_tok = torch.exp(token_log_ratio)
        coef_2_tok = torch.clamp(coef_1_tok, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high)

        per_token_loss1 = coef_1_tok * token_advantages
        per_token_loss2 = coef_2_tok * token_advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2) * token_mask

        loss_per_seq = per_token_loss.sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)
        loss = loss_per_seq.mean()

        if self.beta != 0.0 and "old_ref_per_token_logps" in inputs:
            ref_per_token_logps = inputs["old_ref_per_token_logps"]
            per_token_kl = (per_token_logps - ref_per_token_logps) * token_mask
            kl_per_seq = per_token_kl.sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)
            kl_loss = kl_per_seq.mean()
            loss = loss + self.beta * kl_loss

            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["token_kl"].append(self.accelerator.gather(kl_per_seq).nanmean().item())

        mode = "train" if self.model.training else "eval"

        mean_entropy = (entropies * token_mask).sum() / token_mask.sum().clamp(min=1.0)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1_tok < 1 - self.epsilon_low) & (token_advantages < 0)
        is_high_clipped = (coef_1_tok > 1 + self.epsilon_high) & (token_advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        gathered_low_clip = self.accelerator.gather(is_low_clipped.float().mean())
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())

        gathered_high_clip = self.accelerator.gather(is_high_clipped.float().mean())
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())

        gathered_clip_ratio = self.accelerator.gather(is_region_clipped.float().mean())
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss

    def _token_logps_of_given_y(
        self,
        context_texts: list[str],
        completion_ids_list: list[list[int]],
        ignore_first_k: int = 4,
        model: Optional[nn.Module] = None,
    ):
        device = self.accelerator.device
        model = self.model if model is None else model

        enc = self.processing_class(
            text=context_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,
        )
        enc = BaseTrainer._prepare_inputs(self, enc)

        y_ids = [torch.tensor(ids, device=device, dtype=torch.long) for ids in completion_ids_list]
        y_mask = [torch.ones_like(t, dtype=torch.long) for t in y_ids]
        y_ids = pad(y_ids, padding_value=self.pad_token_id, padding_side="right")
        y_mask = pad(y_mask, padding_value=0, padding_side="right")

        prompt_ids, prompt_mask = enc["input_ids"], enc["attention_mask"]
        input_ids = torch.cat([prompt_ids, y_ids], dim=1)
        attn_mask = torch.cat([prompt_mask, y_mask], dim=1)
        logits_to_keep = y_ids.size(1)

        with torch.no_grad():
            per_tok_logps, _ = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attn_mask,
                logits_to_keep,
                batch_size=self.args.per_device_train_batch_size if self.model.training else self.args.per_device_eval_batch_size,
            )

        if ignore_first_k > 0:
            per_tok_logps = per_tok_logps[:, ignore_first_k:]
            y_ids = y_ids[:, ignore_first_k:]
            y_mask = y_mask[:, ignore_first_k:]

        return per_tok_logps, y_ids, y_mask

    def _sum_logp_of_given_y(
        self,
        context_texts: list[str],
        completion_ids_list: list[list[int]],
        ignore_first_k: int = 0,
    ) -> torch.Tensor:
        per_tok_logps, _, y_mask = self._token_logps_of_given_y(
            context_texts=context_texts,
            completion_ids_list=completion_ids_list,
            ignore_first_k=ignore_first_k,
            model=self.model,
        )
        return (per_tok_logps * y_mask).sum(1)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"

        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}

        BaseTrainer.log(self, logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_token_logps and self._token_logs:
            try:
                with open(self.token_log_file, "a", encoding="utf-8") as f:
                    step = int(self.state.global_step)
                    for rec in list(self._token_logs):
                        json.dump({"step": step, **rec}, f, ensure_ascii=False)
                        f.write("\n")
            except Exception:
                pass
            finally:
                self._token_logs.clear()

    def _generate_single_turn(self, prompts: list[str], images: Optional[list]):
        device = self.accelerator.device

        kwargs = {}
        if images is not None:
            kwargs = {"images": images}
            for prompt, image_list in zip(prompts, images):
                if isinstance(prompt, list):
                    prepare_multimodal_messages(prompt, num_images=len(image_list))

        prompts_text = []
        for prompt in prompts:
            if isinstance(prompt, list):
                text = self.processing_class.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            else:
                text = prompt
            prompts_text.append(text)

        if images is not None:
            prompt_inputs = self.processing_class(text=prompts_text, padding=True, return_tensors="pt", **kwargs)
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up()

            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if images is not None:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    ordered_set_of_images = all_images[:: self.num_generations] if images is not None else None

                    with profiling_context(self, "vLLM.generate"):
                        output = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            truncate_prompt_tokens=self.max_prompt_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                        payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"])
                else:
                    payload = None

                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, _ = obj_list[0]

                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(self.num_generations)]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]

            elif self.vllm_mode == "colocate":
                from vllm import SamplingParams

                if self.guided_decoding_regex:
                    from vllm.sampling_params import GuidedDecodingParams

                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "truncate_prompt_tokens": self.max_prompt_length,
                    "guided_decoding": guided_decoding,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if images is not None:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images

                if images is not None and all_images:
                    vllm_inputs = []
                    for prompt, image_list in zip(all_prompts_text, all_images):
                        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image_list}})
                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)

        elif self.use_transformers_paged:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"

            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)

                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                    )
                    unwrapped_model.train()

            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = paged_prompt_inputs.input_ids
            self.model_wrapped.config._attn_implementation = previous_attn

        else:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            generate_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                **kwargs,
            )
            generate_inputs = BaseTrainer._prepare_inputs(self, generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config, disable_compile=True
                )

            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]

        return prompt_ids, completion_ids, forward_kwargs

    def _distributed_generate_user_feedback(
        self,
        prompts_for_user: list[str],
        completions_for_user: list[str],
    ) -> list[str]:
        if len(prompts_for_user) != len(completions_for_user):
            raise ValueError("prompts/completions length mismatch")

        device = self.accelerator.device
        local_n = len(prompts_for_user)

        local_n_t = torch.tensor([local_n], device=device, dtype=torch.long)
        all_n_t = self.accelerator.gather(local_n_t)
        all_ns = all_n_t.detach().cpu().tolist()

        rank = self.accelerator.process_index
        start = sum(all_ns[:rank])
        end = start + all_ns[rank]
        total = sum(all_ns)

        all_prompts = gather_object(prompts_for_user)
        all_completions = gather_object(completions_for_user)

        if self.accelerator.is_main_process:
            if len(all_prompts) != total or len(all_completions) != total:
                raise RuntimeError(
                    f"Gather mismatch: prompts={len(all_prompts)}, completions={len(all_completions)}, expected={total}"
                )

            all_feedback = self.user_simulator.generate_feedback(
                prompts=all_prompts,
                completions=all_completions,
            )
            if len(all_feedback) != total:
                raise RuntimeError(f"user_simulator returned {len(all_feedback)} feedbacks, expected {total}")
        else:
            all_feedback = None

        obj_list = [all_feedback]
        broadcast_object_list(obj_list, from_process=0)
        all_feedback = obj_list[0]

        return all_feedback[start:end]

    def _maybe_log_token_table(
        self,
        global_indices,
        y_ids,
        y_mask,
        logps_x,
        logps_xo,
        per_tok_signal,
        raw_prompt,
        completion,
        user_response,
        conditional_contexts,
    ):
        if not self.accelerator.is_main_process:
            return

        step = int(self.state.global_step)
        if step % 200 != 0:
            return

        tokenizer = self.processing_class
        max_tokens = 200

        y_ids_cpu = y_ids.detach().cpu()
        y_mask_cpu = y_mask.detach().cpu()
        logps_x_cpu = logps_x.detach().cpu()
        logps_xo_cpu = logps_xo.detach().cpu()
        signal_cpu = per_tok_signal.detach().cpu()

        seen = set()
        for b_idx, g_idx in enumerate(global_indices):
            if g_idx in seen:
                continue
            seen.add(g_idx)

            mask = y_mask_cpu[b_idx].bool()
            if mask.sum().item() == 0:
                continue

            tok_ids = y_ids_cpu[b_idx][mask].tolist()
            lp_x = logps_x_cpu[b_idx][mask].tolist()
            lp_xo = logps_xo_cpu[b_idx][mask].tolist()
            ratios = signal_cpu[b_idx][mask].tolist()

            n = min(len(tok_ids), max_tokens)
            tok_ids, lp_x, lp_xo, ratios = tok_ids[:n], lp_x[:n], lp_xo[:n], ratios[:n]
            tok_strings = [
                tokenizer.decode([tid], clean_up_tokenization_spaces=False).replace("\n", "\\n")
                for tid in tok_ids
            ]

            print(f"\n[SDPOUI DEBUG] step={step}", flush=True)
            print("Full Conditional Prompt:\n" + conditional_contexts, flush=True)
            print("Completion:\n" + completion, flush=True)
            print("User response:\n" + user_response, flush=True)

            header = f"{'idx':>4} | {'tok_id':>7} | {'tok_str':<15} | {'logp_x':>12} | {'logp_xo':>12} | {'log_ratio':>12}"
            print(header, flush=True)
            print("-" * len(header), flush=True)

            for i, (tid, tstr, lx, lxo, r) in enumerate(zip(tok_ids, tok_strings, lp_x, lp_xo, ratios)):
                tstr_display = (tstr[:12] + "…") if len(tstr) > 15 else tstr
                print(
                    f"{i:4d} | {tid:7d} | {tstr_display:<15} | {lx:12.8f} | {lxo:12.8f} | {r:12.8f}",
                    flush=True,
                )

            if mask.sum().item() > max_tokens:
                print(f"... (truncated to first {max_tokens} tokens)", flush=True)
            print("", flush=True)
