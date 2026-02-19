from __future__ import annotations

import json
import re
from datasets import load_dataset

DATASET_NAME = "nvidia/HelpSteer2"
OUT_TRAIN = "train.jsonl"
OUT_VAL = "validation.jsonl"

MAX_PROMPT_CHARS = 2048
MAX_RESPONSE_CHARS = 2048

# Detect extra_id markers that indicate multi-turn / templated conversation format
EXTRA_ID_RE = re.compile(r"<extra_id_\d+>", re.IGNORECASE)


def clean_str(x) -> str:
    return (x or "").strip()


def is_multiturn_prompt(prompt: str) -> bool:
    # Strict: any extra_id marker indicates this is not a clean single-turn prompt.
    return bool(EXTRA_ID_RE.search(prompt))


def save_split(split_name: str, out_path: str) -> None:
    ds = load_dataset(DATASET_NAME, split=split_name)

    written = 0
    scanned = 0
    skipped_missing = 0
    skipped_too_long_prompt = 0
    skipped_too_long_response = 0
    skipped_duplicate_prompt = 0
    skipped_multiturn = 0

    seen_prompts: set[str] = set()

    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            scanned += 1

            prompt = clean_str(row.get("prompt"))
            response = clean_str(row.get("response"))

            # Still require a response so we can filter by response length
            if not prompt or not response:
                skipped_missing += 1
                continue

            # Filter out multi-turn prompts (extra_id markers)
            if is_multiturn_prompt(prompt):
                skipped_multiturn += 1
                continue

            if len(prompt) > MAX_PROMPT_CHARS:
                skipped_too_long_prompt += 1
                continue

            if len(response) > MAX_RESPONSE_CHARS:
                skipped_too_long_response += 1
                continue

            # Unique prompts within this split
            if prompt in seen_prompts:
                skipped_duplicate_prompt += 1
                continue
            seen_prompts.add(prompt)

            # Save prompt only
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n=== Split: {split_name} ===")
    print(f"Scanned rows: {scanned}")
    print(f"Written: {written} -> {out_path}")
    print(f"Skipped missing prompt/response: {skipped_missing}")
    print(f"Skipped multi-turn prompts (extra_id): {skipped_multiturn}")
    print(f"Skipped prompt >{MAX_PROMPT_CHARS} chars: {skipped_too_long_prompt}")
    print(f"Skipped response >{MAX_RESPONSE_CHARS} chars: {skipped_too_long_response}")
    print(f"Skipped duplicate prompts: {skipped_duplicate_prompt}")


def main() -> None:
    save_split("train", OUT_TRAIN)
    save_split("validation", OUT_VAL)


if __name__ == "__main__":
    main()
