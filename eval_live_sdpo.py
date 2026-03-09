"""
Automated evaluation of LiveSDPOUpdater on HelpSteer prompts.

Simulates the live chat flow: for each training prompt, generate a response,
get simulated user feedback, and run one SDPO training step. Periodically
evaluates the current model against the pre-training baseline using a
ClaudeStyleJudge.

Usage:
    python eval_live_sdpo.py \
        --train_jsonl data/helpsteer_prompts/train.jsonl \
        --val_jsonl data/helpsteer_prompts/validation.jsonl \
        --style concise_casual_beginner \
        --run_name my_experiment
"""
import argparse
import copy
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from live_config import LiveSDPOConfig
from live_updater import LiveSDPOUpdater
from auxiliary.claude_user_simulator import ClaudeStyleUserSimulator
from auxiliary.user_simulator import StyleUserSimulator
from auxiliary.claude_style_judge import ClaudeStyleJudge


# ── Metrics (from eval_style_pairwise_accelerate.py) ────────────────────

def _non_tie_outcomes(decisions: List[int]) -> np.ndarray:
    return np.array([1 if d == 0 else 0 for d in decisions if d != -1], dtype=np.int8)


def bootstrap_prop_se(y: np.ndarray, B: int = 10_000, seed: Optional[int] = None) -> float:
    n = int(y.size)
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boot_means = y[idx].mean(axis=1)
    return float(boot_means.std(ddof=1))


def compute_metrics(decisions: List[int], bootstrap_seed: Optional[int] = None) -> Dict:
    n = len(decisions)
    if n == 0:
        return {"n": 0, "coverage": 0.0}

    ties = sum(d == -1 for d in decisions)
    wins_a = sum(d == 0 for d in decisions)
    wins_b = sum(d == 1 for d in decisions)
    n_eff = wins_a + wins_b
    coverage = 1.0 - (ties / n)

    if n_eff == 0:
        return {"n": n, "wins_a": 0, "wins_b": 0, "ties": ties,
                "coverage": coverage, "winrate_a": float("nan"),
                "se": float("nan"), "n_effective": 0}

    p_hat = wins_a / n_eff
    se_analytic = float(np.sqrt(p_hat * (1.0 - p_hat) / n_eff) * 100.0) if n_eff > 1 else 0.0
    se_boot = float(bootstrap_prop_se(_non_tie_outcomes(decisions), seed=bootstrap_seed) * 100.0)

    return {
        "n": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "coverage": float(coverage),
        "winrate_a": float(p_hat * 100.0),
        "se": se_analytic,
        "se_bootstrap": se_boot,
        "n_effective": n_eff,
    }


# ── Evaluation helpers ──────────────────────────────────────────────────

def generate_eval_completions(
    updater: LiveSDPOUpdater,
    eval_messages: List[List[Dict[str, str]]],
    max_new_tokens: int = 512,
) -> List[str]:
    """Generate completions for all eval prompts using the current model (greedy)."""
    original_config = updater.generation_config
    updater.generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=updater.tokenizer.eos_token_id,
        pad_token_id=updater.tokenizer.pad_token_id,
    )
    completions = []
    for messages in eval_messages:
        completions.append(updater.generate_response(messages))
    updater.generation_config = original_config
    return completions


def run_evaluation(
    judge: ClaudeStyleJudge,
    raw_prompts: List[str],
    completions_current: List[str],
    completions_baseline: List[str],
    bootstrap_seed: Optional[int] = None,
) -> Dict:
    """Run pairwise judge comparison, return metrics dict."""
    decisions = judge.choose_batch_generated(
        prompts=raw_prompts,
        completions_a=completions_current,
        completions_b=completions_baseline,
    )
    return compute_metrics(decisions, bootstrap_seed=bootstrap_seed)


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate LiveSDPOUpdater on HelpSteer prompts")

    # Model + training
    p.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--loss_mode", type=str, default="full_distillation",
                    choices=["simple_signal", "full_distillation"])
    p.add_argument("--distillation_topk", type=int, default=50)
    p.add_argument("--use_vllm", action="store_true")

    # Dataset
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--train_n", type=int, default=50)
    p.add_argument("--eval_n", type=int, default=50)
    p.add_argument("--max_prompt_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    # User simulator
    p.add_argument("--style", type=str, default="concise_casual_beginner")
    p.add_argument("--user_model_name_or_path", type=str, default=None,
                    help="If set, use local StyleUserSimulator. Else uses Claude API.")

    # Judge
    p.add_argument("--judge_model", type=str, default="claude-haiku-4-5-20251001")

    # Evaluation schedule
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_max_new_tokens", type=int, default=512)

    # Generation (training)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)

    # Output
    p.add_argument("--out_dir", type=str, default="./live_eval_results")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--wandb", action="store_true")

    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Build config ──
    config = LiveSDPOConfig(
        model_name_or_path=args.model,
        learning_rate=args.lr,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        loss_mode=args.loss_mode,
        distillation_topk=args.distillation_topk,
        use_vllm=args.use_vllm,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        log_to_wandb=args.wandb,
        wandb_project="live-sdpo-eval",
        wandb_run_name=args.run_name,
        checkpoint_every_n_steps=0,  # disable auto-checkpointing
    )

    print(f"[EVAL] Model:      {config.model_name_or_path}", flush=True)
    print(f"[EVAL] LoRA:       {config.use_lora} (r={config.lora_r})", flush=True)
    print(f"[EVAL] LR:         {config.learning_rate}", flush=True)
    print(f"[EVAL] Loss mode:  {config.loss_mode}", flush=True)
    print(f"[EVAL] vLLM:       {config.use_vllm}", flush=True)
    print(f"[EVAL] Style:      {args.style}", flush=True)

    # ── 2. Load datasets ──
    tok = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dsd = load_dataset("json", data_files={"train": args.train_jsonl, "validation": args.val_jsonl})
    train_ds = dsd["train"]
    eval_ds = dsd["validation"]

    def add_len(example):
        messages = [{"role": "user", "content": example["prompt"].strip()}]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        return {"lengths": len(ids)}

    train_ds = train_ds.map(add_len)
    train_ds = train_ds.filter(lambda l: l <= args.max_prompt_tokens, input_columns="lengths").remove_columns("lengths")
    eval_ds = eval_ds.map(add_len)
    eval_ds = eval_ds.filter(lambda l: l <= args.max_prompt_tokens, input_columns="lengths").remove_columns("lengths")

    train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.train_n, len(train_ds))))
    eval_ds = eval_ds.shuffle(seed=args.seed).select(range(min(args.eval_n, len(eval_ds))))

    print(f"[EVAL] Train size: {len(train_ds)}", flush=True)
    print(f"[EVAL] Eval size:  {len(eval_ds)}", flush=True)

    # Pre-build eval messages and raw prompts
    eval_raw_prompts = [ex["prompt"].strip() for ex in eval_ds]
    eval_messages = [
        [{"role": "user", "content": prompt}]
        for prompt in eval_raw_prompts
    ]

    # ── 3. Initialize updater ──
    print("[EVAL] Initializing LiveSDPOUpdater...", flush=True)
    updater = LiveSDPOUpdater(config)

    # ── 4. Initialize user simulator ──
    if args.user_model_name_or_path is not None:
        print(f"[EVAL] Loading local user simulator: {args.user_model_name_or_path}", flush=True)
        user_hf = AutoModelForCausalLM.from_pretrained(
            args.user_model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        user_hf.eval()
        user_tok = AutoTokenizer.from_pretrained(args.user_model_name_or_path, use_fast=True)
        if user_tok.pad_token is None:
            user_tok.pad_token = user_tok.eos_token
        simulator = StyleUserSimulator(
            model=user_hf, tokenizer=user_tok,
            device=next(user_hf.parameters()).device, style=args.style,
        )
    else:
        print("[EVAL] Using Claude user simulator.", flush=True)
        simulator = ClaudeStyleUserSimulator(style=args.style, max_tokens=256, temperature=0.0)

    # ── 5. Initialize judge ──
    print(f"[EVAL] Using ClaudeStyleJudge ({args.judge_model})", flush=True)
    judge = ClaudeStyleJudge(style=args.style, model=args.judge_model)

    # ── 6. Generate baseline eval completions ──
    print(f"[EVAL] Generating baseline completions ({len(eval_messages)} prompts)...", flush=True)
    t0 = time.time()
    baseline_completions = generate_eval_completions(
        updater, eval_messages, max_new_tokens=args.eval_max_new_tokens,
    )
    print(f"[EVAL] Baseline generation took {time.time() - t0:.1f}s", flush=True)

    # ── 7. Training loop ──
    training_metrics = []
    eval_history = []

    print(f"\n[EVAL] Starting training loop ({len(train_ds)} steps)...\n", flush=True)

    for i, example in enumerate(train_ds):
        raw_prompt = example["prompt"].strip()
        messages = [{"role": "user", "content": raw_prompt}]

        # Generate response
        t0 = time.time()
        response = updater.generate_response(messages)
        gen_time = time.time() - t0

        # Get simulated user feedback
        t1 = time.time()
        feedback = simulator.generate_feedback([raw_prompt], [response])[0]
        sim_time = time.time() - t1

        # Train step
        metrics = updater.train_step(
            messages_before_response=messages,
            assistant_response=response,
            user_follow_up=feedback,
        )
        metrics["gen_time_s"] = round(gen_time, 2)
        metrics["sim_time_s"] = round(sim_time, 2)
        training_metrics.append(metrics)

        # Print training metrics
        step = metrics["step"]
        loss = metrics.get("loss", 0.0)
        kl = metrics.get("kl_mean", metrics.get("signal_mean", 0.0))
        train_t = metrics.get("train_time_s", 0.0)
        print(
            f"[STEP {step:4d}] loss={loss:.6f}  kl/signal={kl:.6f}  "
            f"gen={gen_time:.1f}s  sim={sim_time:.1f}s  train={train_t:.1f}s",
            flush=True,
        )

        # Periodic evaluation
        if args.eval_every > 0 and (i + 1) % args.eval_every == 0:
            print(f"\n[EVAL @ step {step}] Generating eval completions...", flush=True)
            t_eval = time.time()
            current_completions = generate_eval_completions(
                updater, eval_messages, max_new_tokens=args.eval_max_new_tokens,
            )
            eval_gen_time = time.time() - t_eval

            print(f"[EVAL @ step {step}] Running judge ({len(eval_raw_prompts)} comparisons)...", flush=True)
            t_judge = time.time()
            eval_metrics = run_evaluation(
                judge, eval_raw_prompts, current_completions,
                baseline_completions, bootstrap_seed=args.seed,
            )
            judge_time = time.time() - t_judge

            eval_metrics["step"] = step
            eval_metrics["eval_gen_time_s"] = round(eval_gen_time, 1)
            eval_metrics["judge_time_s"] = round(judge_time, 1)
            eval_history.append(eval_metrics)

            wr = eval_metrics.get("winrate_a", float("nan"))
            se = eval_metrics.get("se", float("nan"))
            print(
                f"[EVAL @ step {step}] winrate={wr:.1f}% ±{se:.1f}  "
                f"wins={eval_metrics['wins_a']} losses={eval_metrics['wins_b']} "
                f"ties={eval_metrics['ties']}  "
                f"gen={eval_gen_time:.1f}s  judge={judge_time:.1f}s\n",
                flush=True,
            )

    # ── 8. Final evaluation ──
    final_step = updater.step
    if not eval_history or eval_history[-1]["step"] != final_step:
        print(f"\n[EVAL FINAL @ step {final_step}] Generating eval completions...", flush=True)
        t_eval = time.time()
        current_completions = generate_eval_completions(
            updater, eval_messages, max_new_tokens=args.eval_max_new_tokens,
        )
        eval_gen_time = time.time() - t_eval

        print(f"[EVAL FINAL @ step {final_step}] Running judge...", flush=True)
        t_judge = time.time()
        eval_metrics = run_evaluation(
            judge, eval_raw_prompts, current_completions,
            baseline_completions, bootstrap_seed=args.seed,
        )
        judge_time = time.time() - t_judge

        eval_metrics["step"] = final_step
        eval_metrics["eval_gen_time_s"] = round(eval_gen_time, 1)
        eval_metrics["judge_time_s"] = round(judge_time, 1)
        eval_history.append(eval_metrics)

        wr = eval_metrics.get("winrate_a", float("nan"))
        se = eval_metrics.get("se", float("nan"))
        print(
            f"[EVAL FINAL @ step {final_step}] winrate={wr:.1f}% ±{se:.1f}  "
            f"wins={eval_metrics['wins_a']} losses={eval_metrics['wins_b']} "
            f"ties={eval_metrics['ties']}\n",
            flush=True,
        )

    # ── 9. Save results ──
    results = {
        "meta": {
            "model": args.model,
            "style": args.style,
            "loss_mode": args.loss_mode,
            "distillation_topk": args.distillation_topk,
            "learning_rate": args.lr,
            "use_lora": args.use_lora,
            "lora_r": args.lora_r,
            "use_vllm": args.use_vllm,
            "train_n": len(train_ds),
            "eval_n": len(eval_ds),
            "eval_every": args.eval_every,
            "eval_max_new_tokens": args.eval_max_new_tokens,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "judge_model": args.judge_model,
        },
        "eval_history": eval_history,
        "training_metrics": training_metrics,
    }

    out_path = os.path.join(args.out_dir, f"{args.run_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[EVAL] Results saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
