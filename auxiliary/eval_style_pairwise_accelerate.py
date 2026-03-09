# auxiliary/eval_style_pairwise_accelerate.py
from __future__ import annotations

import os
import json
import time
import glob
import argparse
from datetime import timedelta
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from datasets import load_from_disk, load_dataset, Dataset, DatasetDict
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from auxiliary.style_judge import StyleJudge
from auxiliary.claude_style_judge import ClaudeStyleJudge


def parse_args():
    p = argparse.ArgumentParser()

    # p.add_argument("--in_context_evaluation", type=bool, default=False)
    p.add_argument("--in_context_evaluation", action="store_true")

    # Dataset
    p.add_argument("--local_dataset_dir", type=str, required=True)
    p.add_argument("--eval_split", type=str, default="validation")
    p.add_argument("--eval_n", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_prompt_tokens_filter", type=int, default=512)

    # Models to compare
    p.add_argument("--model_a_name_or_path", type=str, required=True)
    p.add_argument("--model_b_name_or_path", type=str, required=True)
    p.add_argument("--tokenizer_a_name_or_path", type=str, default=None)
    p.add_argument("--tokenizer_b_name_or_path", type=str, default=None)

    # Generation
    p.add_argument("--max_input_tokens", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)

    # Judge
    p.add_argument("--judge_model_name_or_path", type=str, required=True)
    p.add_argument("--judge_tokenizer_name_or_path", type=str, default=None)
    p.add_argument("--style", type=str, required=True)
    p.add_argument("--judge_max_input_tokens", type=int, default=2048)
    p.add_argument("--tie_margin", type=float, default=0.1)
    p.add_argument("--judge_batch_size", type=int, default=16)

    # Output
    p.add_argument("--out_dir", type=str, default="style_eval_results")
    p.add_argument("--run_name", type=str, required=True)

    p.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Write summary of the text that is 1-2 sentences long. "
            "Always begin with 'TL;DR:' and output only the summary."
        ),
    )

    return p.parse_args()


TLDR_MARKERS = ["\nTL;DR:\n", "\nTL;DR:", "TL;DR:\n", "TL;DR:"]


def strip_tldr_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.rstrip()
    for m in TLDR_MARKERS:
        if t.endswith(m):
            t = t[: -len(m)].rstrip()
            break
    return t


def _load_any_dataset(path_or_dir: str) -> Any:
    p = Path(path_or_dir)
    if p.is_dir():
        return load_from_disk(path_or_dir)

    # If it's a json/jsonl file, load as a single split dataset.
    suffix = p.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        dsd = load_dataset("json", data_files={"data": str(p)})
        return dsd["data"]

    raise ValueError(f"Unsupported dataset path: {path_or_dir} (not a dir and not .json/.jsonl)")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_get_prompt(example: Dict[str, Any]) -> str:
    # Try common fields; fall back to first user message if available
    for k in ("prompt", "text", "article", "document", "source"):
        if k in example and isinstance(example[k], str) and example[k].strip():
            return example[k]
    msgs = example.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        c = msgs[0].get("content", "")
        if isinstance(c, str):
            return c
    # Last resort: stringify
    return json.dumps(example, ensure_ascii=False)[:5000]


def build_messages(system_prompt: str, user_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def format_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
    if getattr(tokenizer, "apply_chat_template", None) is not None:
        # enable_thinking is model-specific; safe to pass only if accepted.
        # Many tokenizers ignore unknown kwargs, but some will error. Be conservative.
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    # crude fallback
    return messages[-1]["content"] + "\n\nAssistant:"


def _pick_split(ds_any: Any, split: str) -> Dataset:
    if isinstance(ds_any, DatasetDict):
        if split in ds_any:
            return ds_any[split]
        for alt in ("eval", "validation", "val", "test", "train"):
            if alt in ds_any:
                return ds_any[alt]
        raise ValueError(f"Split '{split}' not found in DatasetDict keys={list(ds_any.keys())}")
    if isinstance(ds_any, Dataset):
        return ds_any
    raise TypeError(f"Unsupported dataset type: {type(ds_any)}")


def load_and_prepare_eval_ds(
    accelerator: Accelerator,
    dataset_dir: str,
    split: str,
    eval_n: int,
    seed: int,
    max_prompt_tokens: int,
    filter_tokenizer_name_or_path: str,
    system_prompt: str,
    # Persona is injected ONLY for model A when in_context_evaluation=True.
    system_persona: Optional[str] = None,
) -> Dataset:
    """
    Returns ONE canonical eval dataset (shuffled+subsampled+filtered once) with:
      - global_idx
      - raw_prompt
      - messages_a (system_prompt (+persona optional) + user)
      - messages_b (system_prompt + user)
    Filtering is done ONCE to avoid A/B mismatches.
    By default, filtering uses messages_a if persona is present (strict alignment).
    """
    with accelerator.main_process_first():
        ds_any = _load_any_dataset(dataset_dir)
        eval_ds = _pick_split(ds_any, split)

        # ds_any = load_from_disk(dataset_dir)
        # eval_ds = _pick_split(ds_any, split)

        # stable global_idx before shuffling
        eval_ds = eval_ds.map(lambda x, idx: {"global_idx": idx}, with_indices=True)

        # subsample deterministically
        eval_ds = eval_ds.shuffle(seed=seed).select(range(min(eval_n, len(eval_ds))))

        # Normalize prompt text once
        def add_raw_prompt(ex):
            raw = strip_tldr_text(safe_get_prompt(ex))
            return {"raw_prompt": raw}

        eval_ds = eval_ds.map(add_raw_prompt)

        # Add both message variants
        def add_messages(ex):
            user_text = ex["raw_prompt"]
            msg_b = build_messages(system_prompt, user_text)
            if system_persona is not None:
                sys_a = system_prompt + "\n" + "The user you are interacting with has the following persona:\n" + system_persona
                msg_a = build_messages(sys_a, user_text)
            else:
                msg_a = msg_b
            return {"messages_a": msg_a, "messages_b": msg_b}

        eval_ds = eval_ds.map(add_messages)

        # Length filter using a reference tokenizer (deterministic).
        # IMPORTANT: To preserve strict A/B alignment, filter on the *longer* variant:
        # - if persona exists => messages_a
        # - else => messages_b (same anyway)
        tok = AutoTokenizer.from_pretrained(filter_tokenizer_name_or_path, trust_remote_code=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        key_for_filter = "messages_a" if system_persona is not None else "messages_b"

        def add_len(ex):
            text = format_messages(tok, ex[key_for_filter])
            ids = tok(text, add_special_tokens=False).input_ids
            return {"lengths": len(ids)}

        eval_ds = eval_ds.map(add_len)
        eval_ds = eval_ds.filter(lambda l: l <= max_prompt_tokens, input_columns="lengths").remove_columns("lengths")

    return eval_ds


@torch.no_grad()
def generate_for_dataset(
    accelerator: Accelerator,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    local_ds: Dataset,
    messages_key: str,
    max_input_tokens: int,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
) -> Dict[int, str]:
    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(accelerator.device)
    model.eval()

    outputs: Dict[int, str] = {}

    for start in range(0, len(local_ds), batch_size):
        batch = local_ds.select(range(start, min(len(local_ds), start + batch_size)))
        messages = [ex[messages_key] for ex in batch]
        prompts = [format_messages(tok, m) for m in messages]

        if start == 0 and accelerator.is_main_process:
            print(f"\n\nPrompt Example ({messages_key}):", prompts[:], flush=True)

        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=False,
        ).to(accelerator.device)

        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        gen = model.generate(**enc, **gen_kwargs)

        # decode only generated part (LEFT-PAD SAFE)
        base_len = enc["input_ids"].shape[1]  # padded prompt length for the whole batch
        for i, ex in enumerate(batch):
            glb = int(ex["global_idx"])
            gen_ids = gen[i, base_len:]
            out = tok.decode(gen_ids, skip_special_tokens=True).strip()
            outputs[glb] = out

    # free ASAP
    del model
    torch.cuda.empty_cache()

    return outputs




# helpers
def _non_tie_outcomes(decisions: List[int]) -> np.ndarray:
    """
    Returns array of length n_eff with values:
      1 for A win (decision==0)
      0 for B win (decision==1)
    Ties (-1) are removed.
    """
    return np.array([1 if d == 0 else 0 for d in decisions if d != -1], dtype=np.int8)


def bootstrap_prop_se(y: np.ndarray, B: int = 10_000, seed: Optional[int] = None) -> float:
    """
    Bootstrap SE for a proportion mean(y), where y in {0,1}.
    Returns SE in [0,1] units.
    """
    n = int(y.size)
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0

    rng = np.random.default_rng(seed)
    # Vectorized bootstrap: sample indices shape (B, n)
    idx = rng.integers(0, n, size=(B, n))
    boot_means = y[idx].mean(axis=1)
    return float(boot_means.std(ddof=1))


def compute_metrics(
    decisions: List[int],
    *,
    bootstrap: bool = True,
    bootstrap_B: int = 10_000,
    bootstrap_seed: Optional[int] = None,
) -> Dict[str, Any]:
    n = len(decisions)
    if n == 0:
        return {"n": 0, "coverage": 0.0}

    ties = sum(d == -1 for d in decisions)
    wins_a = sum(d == 0 for d in decisions)
    wins_b = sum(d == 1 for d in decisions)

    n_eff = wins_a + wins_b
    coverage = 1.0 - (ties / n)

    if n_eff == 0:
        p_hat = float("nan")
        se_analytic_pct = float("nan")
        se_boot_pct = float("nan")
        winrate_pct = float("nan")
    else:
        p_hat = wins_a / n_eff  # in [0,1]
        winrate_pct = float(p_hat * 100.0)

        # Analytic (binomial) SE of proportion
        se_analytic_pct = float(np.sqrt(p_hat * (1.0 - p_hat) / n_eff) * 100.0) if n_eff > 1 else 0.0

        # Bootstrap SE of proportion
        if bootstrap:
            y = _non_tie_outcomes(decisions)  # length n_eff, in {0,1}
            se_boot_pct = float(bootstrap_prop_se(y, B=bootstrap_B, seed=bootstrap_seed) * 100.0)
        else:
            se_boot_pct = float("nan")

    return {
        "n": n,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "coverage": float(coverage),

        # Pure winrate ignoring ties (%)
        "winrate_a_ignoring_ties": winrate_pct,
        "n_effective_no_ties": n_eff,

        # Error bars for that pure winrate
        "winrate_a_ignoring_ties_se": se_analytic_pct,
        "winrate_a_ignoring_ties_bootstrap_se": se_boot_pct,
    }


def main():
    args = parse_args()

    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[pg_kwargs])

    rank = accelerator.process_index
    world = accelerator.num_processes
    local_rank = accelerator.local_process_index

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.run_name}.json")
    part_path = out_path.replace(".json", f".rank{rank}.jsonl")

    if accelerator.is_main_process:
        print(f"[{now_ts()}] world_size={world}", flush=True)
        print(f"[{now_ts()}] out_path={out_path}", flush=True)

    # Use model A tokenizer as the reference for filtering, unless you pass something else.
    filter_tok = args.tokenizer_a_name_or_path or args.model_a_name_or_path

    # --- Judge ---

    # --- Judge ---
    use_claude_judge = args.judge_model_name_or_path.lower().startswith("claude-")

    if use_claude_judge:
        judge = ClaudeStyleJudge(
            style=args.style,
            model=args.judge_model_name_or_path,  # e.g. "claude-haiku-4-5-20251001"
            max_tokens=16,
            temperature=0.0,
        )
    else:
        judge_tok = AutoTokenizer.from_pretrained(
            args.judge_tokenizer_name_or_path or args.judge_model_name_or_path,
            trust_remote_code=True,
        )
        judge_tok.padding_side = "left"
        if judge_tok.pad_token is None:
            judge_tok.pad_token = judge_tok.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if args.judge_model_name_or_path == "Qwen/Qwen3-14B":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            judge_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-14B",
                quantization_config=bnb_config,
                device_map="auto",
            )
            judge_model.eval()
            print("Using Quantized Judge Model (Qwen3-14B)")
        else:
            judge_model = AutoModelForCausalLM.from_pretrained(
                args.judge_model_name_or_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(accelerator.device)
            judge_model.eval()

        judge = StyleJudge(
            model=judge_model,
            tokenizer=judge_tok,
            device=accelerator.device,
            style=args.style,
            max_input_tokens=args.judge_max_input_tokens,
            tie_margin=args.tie_margin,
        )

    # Persona only used in the in-context variant, and only for model A messages.
    if args.in_context_evaluation and hasattr(judge, "get_system_persona"):
        system_persona = judge.get_system_persona()
    elif args.in_context_evaluation:
        # Claude judge always uses persona internally; for dataset persona injection,
        # just reuse STYLE_PERSONAS directly
        from auxiliary.user_simulator import STYLE_PERSONAS
        system_persona = STYLE_PERSONAS[args.style]
    else:
        system_persona = None

    # system_persona = judge.get_system_persona() if args.in_context_evaluation else None

    # --- Build ONE canonical eval dataset (no A/B mismatch possible) ---
    eval_ds = load_and_prepare_eval_ds(
        accelerator=accelerator,
        dataset_dir=args.local_dataset_dir,
        split=args.eval_split,
        eval_n=args.eval_n,
        seed=args.seed,
        max_prompt_tokens=args.max_prompt_tokens_filter,
        filter_tokenizer_name_or_path=filter_tok,
        system_prompt=args.system_prompt,
        system_persona=system_persona,
    )

    # Deterministic sharding by position (shared for both models)
    indices = list(range(rank, len(eval_ds), world))
    local_ds = eval_ds.select(indices)

    if accelerator.is_main_process:
        print(f"[{now_ts()}] eval_ds={len(eval_ds)} local_shard~{len(local_ds)}", flush=True)
        print(f"[{now_ts()}] in_context_evaluation={bool(args.in_context_evaluation)} persona_in_A={system_persona is not None}", flush=True)

    # --- Generate outputs for model A ---
    out_a = generate_for_dataset(
        accelerator=accelerator,
        model_name_or_path=args.model_a_name_or_path,
        tokenizer_name_or_path=args.tokenizer_a_name_or_path or args.model_a_name_or_path,
        local_ds=local_ds,
        messages_key="messages_a",
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # --- Generate outputs for model B ---
    out_b = generate_for_dataset(
        accelerator=accelerator,
        model_name_or_path=args.model_b_name_or_path,
        tokenizer_name_or_path=args.tokenizer_b_name_or_path or args.model_b_name_or_path,
        local_ds=local_ds,
        messages_key="messages_b",
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Align by the SAME local_ds ordering
    raw_prompts: List[str] = [ex["raw_prompt"] for ex in local_ds]
    ga: List[str] = [out_a[int(ex["global_idx"])] for ex in local_ds]
    gb: List[str] = [out_b[int(ex["global_idx"])] for ex in local_ds]

    decisions = judge.choose_batch_generated(
        prompts=raw_prompts,
        completions_a=ga,
        completions_b=gb,
        batch_size=args.judge_batch_size,
    )

    # write per-rank jsonl
    with open(part_path, "w") as f:
        for ex, a, b, d in zip(local_ds, ga, gb, decisions):
            f.write(
                json.dumps(
                    {
                        "global_idx": int(ex["global_idx"]),
                        "raw_prompt": ex["raw_prompt"],
                        "output_a": a,
                        "output_b": b,
                        "decision": int(d),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    accelerator.wait_for_everyone()

    # merge + metrics
    if accelerator.is_main_process:
        rows = []
        for pf in sorted(glob.glob(out_path.replace(".json", ".rank*.jsonl"))):
            with open(pf, "r") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))

        rows.sort(key=lambda r: r["global_idx"])
        decisions_all = [r["decision"] for r in rows]

        report = {
            "meta": {
                "run_name": args.run_name,
                "timestamp": now_ts(),
                "dataset_dir": args.local_dataset_dir,
                "eval_split": args.eval_split,
                "eval_n_requested": args.eval_n,
                "seed": args.seed,
                "max_prompt_tokens_filter": args.max_prompt_tokens_filter,
                "model_a": args.model_a_name_or_path,
                "model_b": args.model_b_name_or_path,
                "judge_model": args.judge_model_name_or_path,
                "style": args.style,
                "tie_margin": args.tie_margin,
                "in_context_evaluation": bool(args.in_context_evaluation),
                "gen": {
                    "max_input_tokens": args.max_input_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "batch_size": args.batch_size,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
            },
            "metrics": compute_metrics(
                decisions_all,
                bootstrap=True,              # or False if you only want analytic SE
                bootstrap_B=5000,          # good default for ~256 trials
                bootstrap_seed=args.seed,    # deterministic across runs
            ),
            # "metrics": compute_metrics(decisions_all),
            "examples": rows,
        }

        with open(out_path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        m = report["metrics"]
        print(
            f"[{now_ts()}] DONE  "
            f"winrate(A,no ties)={m['winrate_a_ignoring_ties']:.4f} "
            f"±{m['winrate_a_ignoring_ties_se']:.4f} (analytic SE) "
            f"±{m['winrate_a_ignoring_ties_bootstrap_se']:.4f} (bootstrap SE) "
            f"coverage={m['coverage']:.4f}  "
            f"(wins_a={m['wins_a']} wins_b={m['wins_b']} ties={m['ties']} "
            f"n_eff={m['n_effective_no_ties']} n={m['n']})",
            flush=True,
        )

        # print(
        #     f"[{now_ts()}] DONE  "
        #     f"winrate(A)={m['winrate_a_ignoring_ties']:.4f}  "
        #     f"standard_win_rate={m['standard_win_rate']:.4f}±{m['standard_error']:.4f}  "
        #     f"coverage={m['coverage']:.4f}  "
        #     f"(wins_a={m['wins_a']} wins_b={m['wins_b']} ties={m['ties']} n={m['n']})",
        #     flush=True,
        # )

        # cleanup parts (optional)
        for pf in glob.glob(out_path.replace(".json", ".rank*.jsonl")):
            try:
                os.remove(pf)
            except OSError:
                pass

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
