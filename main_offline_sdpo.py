# Offline SDPO from User Interactions
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from offline_sdpo_trainer import OfflineSDPOCollator, OfflineSDPOTrainer
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--base_model", type=str, default=None)
    p.add_argument("--train_jsonl", type=str, required=True,
                   help="Path to training JSONL (e.g. wildfeedback_interactions.jsonl)")
    p.add_argument("--num_epochs", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    model_name_or_path = args.base_model
    output_dir = os.environ.get("OUTPUT_DIR", "./local_checkpoints")

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    max_completion_len = 2048
    num_epochs = args.num_epochs

    print("Config:")
    print(f"Model:      {model_name_or_path}")
    print(f"Output dir: {output_dir}")
    print(f"Train JSONL:{args.train_jsonl}")
    print(f"LR:         {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Grad accum: {grad_accum}")
    print(f"Epochs:     {num_epochs}")

    print(f"Loading data from {args.train_jsonl}...")
    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    train_dataset = dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    # Not used for generation in offline training, but required by some model configs
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    # for Llama as they don't have a pad token by default
    if model_name_or_path == "meta-llama/Meta-Llama-3.1-8B-Instruct" and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    print("pad_token_id:", tokenizer.pad_token_id, "pad_token:", tokenizer.pad_token)
    print("eos_token_id:", tokenizer.eos_token_id, "eos_token:", tokenizer.eos_token)

    collator = OfflineSDPOCollator(
        tokenizer=tokenizer,
        max_completion_length=max_completion_len
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,

        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",

        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        report_to=["wandb"],

        warmup_ratio=0.05,
        max_grad_norm=10.0,
        lr_scheduler_type="cosine",

        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=42,
    )

    # KL regularization is not used (kl_beta=0, ref_model=None).
    trainer = OfflineSDPOTrainer(
        ignore_first_k=2,
        model=model,
        ref_model=None,
        kl_beta=0,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    print("Training complete.")


if __name__ == "__main__":
    main()
