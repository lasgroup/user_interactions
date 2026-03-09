import os
import argparse
from dataclasses import dataclass
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from online_sdpo_config import SDPOConfig
from online_sdpo_trainer import SDPOOnlineTrainer
from auxiliary.user_simulator import StyleUserSimulator
from auxiliary.claude_user_simulator import ClaudeStyleUserSimulator


SYSTEM_PROMPT_TLDR = (
    "Write summary of the text that is 1-2 sentences long. Always begin with 'TL;DR:' and output only the summary."
)
SYSTEM_PROMPT_GENERAL = ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--style", type=str, default="concise_casual_beginner")
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-8B")
    p.add_argument(
        "--system_prompt",
        type=str,
        default="general",
        choices=["tldr", "general"],
        help="Choose which system prompt experiment to run.",
    )
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--max_prompt_tokens", type=int, default=512)
    p.add_argument("--train_n", type=int, default=512)
    p.add_argument("--eval_n", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--user_model_name_or_path",
        type=str,
        default=None,
        help="If set, use a local StyleUserSimulator with this model. "
             "If unset, ClaudeStyleUserSimulator is used (requires ANTHROPIC_API_KEY).",
    )
    return p.parse_args()


@dataclass
class ScriptArgs:
    model_name_or_path: str = "Qwen/Qwen3-8B"
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    dataset_train_split: str = "train"
    dataset_test_split: str = "validation"


def dummy_reward(prompts, completions, **kwargs):
    return [0.0] * len(completions)


def strip_tldr_suffix(prompt: str) -> str:
    s = prompt.rstrip()
    for m in ["\nTL;DR:\n", "\nTL;DR:", "TL;DR:\n", "TL;DR:"]:
        if s.endswith(m):
            return s[: -len(m)].rstrip()
    return s


def main():
    cli = parse_args()
    args = ScriptArgs(model_name_or_path=cli.model_name_or_path)

    if cli.system_prompt == "tldr":
        system_prompt = SYSTEM_PROMPT_TLDR
    else:
        system_prompt = SYSTEM_PROMPT_GENERAL

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dsd = load_dataset(
        "json",
        data_files={"train": cli.train_jsonl, "validation": cli.val_jsonl},
    )

    train_ds = dsd[args.dataset_train_split]
    eval_ds = dsd[args.dataset_test_split]

    def to_chat(example):
        raw = strip_tldr_suffix(example["prompt"]).strip()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw},
        ]
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return {"prompt": rendered, "raw_prompt": raw}

    def add_len(example):
        ids = tok(example["prompt"], add_special_tokens=False)["input_ids"]
        return {"lengths": len(ids)}

    train_ds = train_ds.map(lambda x, idx: {"global_idx": idx}, with_indices=True)
    eval_ds = eval_ds.map(lambda x, idx: {"global_idx": idx}, with_indices=True)

    train_ds = train_ds.map(to_chat).map(add_len)
    train_ds = train_ds.filter(lambda l: l <= cli.max_prompt_tokens, input_columns="lengths").remove_columns("lengths")

    eval_ds = eval_ds.map(to_chat).map(add_len)
    eval_ds = eval_ds.filter(lambda l: l <= cli.max_prompt_tokens, input_columns="lengths").remove_columns("lengths")

    train_ds = train_ds.shuffle(seed=cli.seed).select(range(min(cli.train_n, len(train_ds))))
    eval_ds = eval_ds.shuffle(seed=cli.seed).select(range(min(cli.eval_n, len(eval_ds))))

    print("Train size:", len(train_ds))
    print("Eval size:", len(eval_ds))
    print("STYLE:", cli.style)
    print("SYSTEM_PROMPT:", cli.system_prompt)

    training_args = SDPOConfig(
        output_dir=args.output_dir,
        model_init_kwargs={"dtype": torch.bfloat16},
        learning_rate=cli.learning_rate,
        lr_scheduler_type="constant",
        max_prompt_length=2048,
        max_completion_length=2048,
        num_generations=1,
        gradient_accumulation_steps=cli.gradient_accumulation_steps,
        per_device_train_batch_size=cli.per_device_train_batch_size,
        steps_per_generation=1,
        use_vllm=False,
        remove_unused_columns=False,
        log_completions=True,
        num_completions_to_print=16,
        wandb_log_unique_prompts=True,
        eval_steps=10,
        logging_steps=3,
        beta=0.0,
        temperature=1.0,
        seed=42,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=3,
        num_train_epochs=1,
        report_to=["wandb"],
    )
    training_args.num_iterations = 1
    training_args.style = cli.style
    training_args.system_prompt = system_prompt

    if cli.user_model_name_or_path is not None:
        print(f"Loading local user simulator: {cli.user_model_name_or_path}")
        user_hf = AutoModelForCausalLM.from_pretrained(
            cli.user_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        user_hf.eval()
        user_sim_tok = AutoTokenizer.from_pretrained(cli.user_model_name_or_path, use_fast=True)
        if user_sim_tok.pad_token is None:
            user_sim_tok.pad_token = user_sim_tok.eos_token
        user_model = StyleUserSimulator(
            model=user_hf,
            tokenizer=user_sim_tok,
            device=next(user_hf.parameters()).device,
            style=cli.style,
        )
    else:
        print("Using Claude user simulator (ClaudeStyleUserSimulator).")
        user_model = ClaudeStyleUserSimulator(
            style=training_args.style,
            max_tokens=256,
            temperature=0.0,
        )

    trainer = SDPOOnlineTrainer(
        model=args.model_name_or_path,
        args=training_args,
        reward_funcs=[dummy_reward],
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=None,
        user_model=user_model,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
