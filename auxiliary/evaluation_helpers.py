import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
import gc
import pandas as pd

@torch.no_grad()
def calculate_kl_divergence(
    new_model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    device: str = "cuda",
    batch_size: int = 4
) -> float:
    """Approximate KL divergence (New || Ref) just based on prompts, no generation."""
    
    new_model.eval()
    ref_model.eval()
    
    total_kl_sum = 0.0
    total_valid_tokens = 0
    
    # Process in chunks
    for i in tqdm(range(0, len(prompts), batch_size), desc="Calculating KL"):
        batch_prompts = prompts[i : i + batch_size]
        
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(device)
        
        # Forward pass - New Model
        new_outputs = new_model(**inputs)
        new_logprobs = F.log_softmax(new_outputs.logits, dim=-1)

        # Forward pass - Ref Model
        ref_outputs = ref_model(**inputs)
        ref_logprobs = F.log_softmax(ref_outputs.logits, dim=-1)

        # KL Calculation
        # KL(P || Q) = sum( P(x) * (log P(x) - log Q(x)) )
        new_probs = torch.exp(new_logprobs)
        log_ratio = new_logprobs - ref_logprobs
        kl_divergence_per_token = new_probs * log_ratio
        
        # Mask padding
        # Sum over vocab first (dim=-1), then multiply by mask
        kl_per_token_masked = kl_divergence_per_token.sum(dim=-1) * inputs.attention_mask
        
        # Accumulate sums
        total_kl_sum += kl_per_token_masked.sum().item()
        total_valid_tokens += inputs.attention_mask.sum().item()
        
        # Cleanup to free VRAM immediately
        del inputs, new_outputs, ref_outputs, new_logprobs, ref_logprobs, new_probs, log_ratio, kl_divergence_per_token
        torch.cuda.empty_cache()

    # Avoid division by zero
    if total_valid_tokens == 0:
        return 0.0
        
    return total_kl_sum / total_valid_tokens


def normalize_wildchat_format(messages):
    normalized = []
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    
    for msg in messages:
        if "value" in msg and "content" not in msg:
            original_role = msg.get("from", "user")
            new_role = role_map.get(original_role, original_role)
            normalized.append({
                "role": new_role, 
                "content": msg["value"]
            })
        else:
            normalized.append(msg)
    return normalized

def run_qualitative_evaluation(
    new_model: PreTrainedModel,
    ref_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_data: Dataset, 
    num_samples: int = 5,
    max_new_tokens: int = 128,
    device: str = "cuda"
):
    print("\n\n=== Qualitative Completion Comparison ===")
    
    if len(eval_data) > num_samples:
        samples = eval_data.shuffle(seed=12345).select(range(num_samples))
    else:
        samples = eval_data

    new_model.eval()
    ref_model.eval()

    results = []

    for i, row in enumerate(tqdm(samples, desc="Generating completions")):
        
        # 1. Prepare Prompt
        raw_prompt_messages = row["prompt"] 
        clean_messages = normalize_wildchat_format(raw_prompt_messages)
        
        prompt_text = tokenizer.apply_chat_template(
            clean_messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False,
        )
        
        inputs = tokenizer(
            prompt_text, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        input_len = inputs['input_ids'].shape[1]

        # 2. Generate New (Checkpoint)
        new_gen_ids = new_model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_len = new_gen_ids.shape[1] - input_len
        new_completion = tokenizer.decode(new_gen_ids[0, input_len:], skip_special_tokens=True)

        # 3. Generate Ref (Baseline)
        ref_gen_ids = ref_model.generate(**inputs, max_new_tokens=max_new_tokens)
        ref_len = ref_gen_ids.shape[1] - input_len
        ref_completion = tokenizer.decode(ref_gen_ids[0, input_len:], skip_special_tokens=True)
        
        # 4. Process Ground Truth (Dataset)
        comp_data = row["completion"]
        if isinstance(comp_data, dict):
            dataset_completion = comp_data.get("value") or comp_data.get("content")
        else:
            dataset_completion = str(comp_data)
            
        # Tokenize GT to get length
        gt_ids = tokenizer(dataset_completion, add_special_tokens=False)["input_ids"]
        gt_len = len(gt_ids)

        # 5. Store Results (Reordered for readability)
        results.append({
            "Prompt_Summary": prompt_text[-150:], # Show end of prompt
            
            "GT_Len": gt_len,
            "Dataset_GPT": dataset_completion[:300] + "..." if len(dataset_completion) > 300 else dataset_completion,
            
            "Ref_Len": ref_len,
            "Ref_Output": ref_completion,
            
            "New_Len": new_len,
            "New_Output": new_completion,
        })
        
        # Cleanup
        del inputs, new_gen_ids, ref_gen_ids
        torch.cuda.empty_cache()
    
    df = pd.DataFrame(results)

    print("\n=== Average Completion Lengths ===")
    print(f"Dataset GPT: {df['GT_Len'].mean():.1f} tokens")
    print(f"Ref Model:   {df['Ref_Len'].mean():.1f} tokens")
    print(f"New Model:   {df['New_Len'].mean():.1f} tokens")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 180)
    pd.set_option('display.max_colwidth', 30) # slightly wider text
    
    print(df.to_markdown(index=False))
