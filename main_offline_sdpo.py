import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed
)
import wandb
from pathlib import Path
from lras_offline_trainer import OfflineLRASCollator, OfflineLRASTrainer
import argparse 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--signal_clip", type=float, default=100.0)
    p.add_argument("--base_model", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()

    model_name_or_path = args.base_model
    
    # data_path = str((Path.home() / "test-time-alignment" / "data" / "wildchat_interactions.jsonl").resolve())
    data_path = str((Path.home() / "test-time-alignment" / "data" / "wildfeedback_interactions.jsonl").resolve())


    output_dir = os.environ.get("OUTPUT_DIR", "./local_checkpoints")

    learning_rate = args.learning_rate    # 1e-6 for Qwen3-4B
    batch_size = args.batch_size          # with gradient checkpointing 4-8 fits for Qwen3-4B; 2-16 fits for Qwen3-8B
    grad_accum = args.grad_accum          
    max_completion_len = 2048
    num_epochs = 2
    
    print("Config:")
    print(f"Model: {model_name_or_path}")
    print(f"Output dir: {output_dir}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size:    {batch_size}")
    print(f"Grad accum:    {grad_accum}")

    # Initialize W&B
    # wandb.init(project="wildfeedback-lras", name="Qwen-4B-2507-5e6-bs-4-no-clip")

    print(f"Loading data from {data_path}...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    

    # dataset = dataset.train_test_split(test_size=0.05)
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["test"]
    train_dataset = dataset # Using full set for now

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    # Doesn't matter here bc we are not on-policy generating, but need to set it for Olmo3 models still.
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    # # Load Ref Model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    ref_model.eval()

    # for Llama as they don't have a pad token by default
    if model_name_or_path == "meta-llama/Meta-Llama-3.1-8B-Instruct" and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    print("pad_token_id:", tokenizer.pad_token_id, "pad_token:", tokenizer.pad_token)
    print("eos_token_id:", tokenizer.eos_token_id, "eos_token:", tokenizer.eos_token)


    collator = OfflineLRASCollator(
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
        # report_to="wandb",
        # run_name="Qwen3-8B-5e6-2-32-sh-cosine-oldct",

        # 14/12/2025: added warm up schedule to learning rate    
        warmup_ratio=0.05, 
        # 14/12/2025: added max grad norm, default is afaik 1.0
        max_grad_norm=10.0,
        
        lr_scheduler_type="cosine", 

        remove_unused_columns=False, 
        dataloader_num_workers=4,
        seed=42,
    )

    # --- 5. Initialize Trainer ---
    trainer = OfflineLRASTrainer(
        signal_clip=args.signal_clip,      
        ignore_first_k=2,      
        model=model,
        ref_model=None,
        kl_beta=0,
        kl_max_new_tokens=256,
        kl_do_sample=True,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer, 
        data_collator=collator,
    )

    trainer.train()
    
    # trainer.save_model(os.path.join(output_dir, "final_model"))
    # tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    print("Training complete.")

if __name__ == "__main__":
    main()