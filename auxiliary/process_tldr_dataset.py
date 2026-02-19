from pathlib import Path
import hashlib
from datasets import load_dataset, DatasetDict, Features, Value, concatenate_datasets

# --- Config ---
ADD_TLDR_SUFFIX = True
N_EXAMPLES_TO_PRINT = 5
TARGET_SUBDIR = "tldr_prompts_unique"
NORMALIZE_FOR_DEDUP = True     # whitespace + lowercase
CROSS_SPLIT_DEDUP = True       # ensure val has no items that appear in train
MAX_PROMPT_LEN = 1024          

def norm_text(s: str) -> str:
    s = s.strip()
    if NORMALIZE_FOR_DEDUP:
        # collapse whitespace and lowercase for stable keys
        s = " ".join(s.split()).lower()
    return s

def make_key(prompt: str) -> str:
    return hashlib.sha1(norm_text(prompt).encode("utf-8")).hexdigest()

# --- Load ---
ds = load_dataset("openai/summarize_from_feedback", "comparisons")  # train/validation


def to_prompt(example):
    info = example.get("info") or {}
    prompt = (info.get("post") or info.get("article") or "").strip()
    if not prompt:
        return {"prompt": "", "keep": False}
    if ADD_TLDR_SUFFIX:
        prompt = prompt + "\nTL;DR:\n"

    if len(prompt) >= MAX_PROMPT_LEN:
        return {"prompt": "", "keep": False}

    return {"prompt": prompt, "keep": True}

def add_key(batch):
    # batched to avoid Python overhead
    prompts = batch["prompt"]
    return {"key": [make_key(p) for p in prompts]}

def mark_first_occurrences(batch, _seen=set()):
    keys = batch["key"]
    keep = []
    for k in keys:
        if k in _seen:
            keep.append(False)
        else:
            _seen.add(k)
            keep.append(True)
    return {"keep": keep}

def dedupe(ds_split):
    # 1) add stable key
    ds_split = ds_split.map(add_key, batched=True, num_proc=1, desc="hashing")
    # 2) keep first occurrence deterministically
    flagged = ds_split.map(
        mark_first_occurrences,
        batched=True,
        num_proc=1,                 # preserve order
        load_from_cache_file=False, # ensure re-run recomputes
        desc="marking first occurrences",
    )
    deduped = flagged.filter(lambda k: k, input_columns=["keep"]).remove_columns("keep")
    return deduped

# --- Build prompts per split ---
train = ds["train"].map(to_prompt, remove_columns=ds["train"].column_names)
train = train.filter(lambda ex: ex["keep"]).remove_columns("keep")
valid = ds["validation"].map(to_prompt, remove_columns=ds["validation"].column_names)
valid = valid.filter(lambda ex: ex["keep"]).remove_columns("keep")

# Enforce schema
features = Features({"prompt": Value("string")})
train = train.cast(features)
valid = valid.cast(features)

print(f"Raw counts -> train: {len(train):,}, valid: {len(valid):,}")

# --- Deduplicate within each split ---
train = dedupe(train)
valid = dedupe(valid)
print(f"After in-split dedupe -> train: {len(train):,}, valid: {len(valid):,}")

# --- Optional: remove validation items that also appear in train ---
if CROSS_SPLIT_DEDUP:
    train_keys = set(train["key"])
    def not_in_train(batch):
        return {"keep": [k not in train_keys for k in batch["key"]]}
    valid = valid.map(not_in_train, batched=True, num_proc=1, desc="cross-split check")
    valid = valid.filter(lambda k: k, input_columns=["keep"]).remove_columns("keep")
    print(f"After cross-split dedupe -> train: {len(train):,}, valid: {len(valid):,}")

# Clean up helper column before saving
train = train.remove_columns("key")
valid = valid.remove_columns("key")

prompt_ds = DatasetDict(train=train, validation=valid)
print(prompt_ds)

# Print a few deterministic examples
k = min(N_EXAMPLES_TO_PRINT, len(prompt_ds["train"]))
for i in range(k):
    ex = prompt_ds["train"][i]["prompt"]
    print(f"\n=== Example {i+1}/{k} ===\n{ex[:1600]}")

# Save to disk + JSONL export
target_dir = (Path.cwd().parent / "data" / TARGET_SUBDIR).resolve()
target_dir.mkdir(parents=True, exist_ok=True)
prompt_ds.save_to_disk(str(target_dir))
print(f"\nSaved HF dataset to: {target_dir}")

for split in ["train", "validation"]:
    out_jsonl = target_dir / f"{split}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        import json
        for row in prompt_ds[split]:
            f.write(json.dumps({"prompt": row["prompt"]}, ensure_ascii=False) + "\n")
    print(f"Exported {split} to: {out_jsonl}")
