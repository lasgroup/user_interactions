# wildfeedback_interactions_short_history.py

from pathlib import Path
from datasets import load_dataset
import pandas as pd

# --- Configuration ---
MAX_TOTAL_CONV_LENGTH = 100000     # Drop if whole conversation > this many chars
MAX_COMPLETION_LENGTH = 4096       # Drop if ANY GPT response > this many chars
MAX_HISTORY_MESSAGES = 5          # Max number of messages in prompt history
OUTPUT_FILENAME = "wildfeedback_interactions.jsonl"


# MIGHT WANT TO CHANGE IT TO MAX_HISTORY_MESSAGES = 5 ----- 14/12/2025

def truncate_history_starting_with_human(history, max_messages):
    """
    Truncate history to at most `max_messages` messages such that:
    - history starts with a human message
    - messages alternate roles
    - empty messages are removed
    Returns None if no valid history remains.
    """
    if not history:
        return None

    # Remove empty messages
    cleaned = [
        m for m in history
        if (m.get("value") or "").strip()
        and m.get("from") in {"human", "gpt"}
    ]

    if not cleaned:
        return None

    # Take the last max_messages messages
    truncated = cleaned[-max_messages:]

    # Drop leading GPT messages until history starts with human
    while truncated and truncated[0]["from"] != "human":
        truncated = truncated[1:]

    if not truncated:
        return None

    # Enforce alternation
    normalized = [truncated[0]]
    for m in truncated[1:]:
        if m["from"] != normalized[-1]["from"]:
            normalized.append(m)

    # Final check
    if not normalized or normalized[0]["from"] != "human":
        return None

    return normalized


print("Loading dataset...")
dataset = load_dataset("microsoft/WildFeedback", "wildfeedback", split="train")

processed_data = []
stats = {
    "total_conversations": 0,
    "kept_conversations": 0,
    "skipped_total_too_long": 0,
    "skipped_completion_too_long": 0,
    "skipped_too_short_or_empty": 0,
    "skipped_no_valid_history": 0,
}

print(
    f"Processing...\n"
    f"  Max total conv length: {MAX_TOTAL_CONV_LENGTH}\n"
    f"  Max completion length: {MAX_COMPLETION_LENGTH}\n"
    f"  Max history messages:  {MAX_HISTORY_MESSAGES}\n"
)

for original_idx, row in enumerate(dataset):
    stats["total_conversations"] += 1

    conversation = row.get("conversations") or row.get("conversation")

    if not conversation or len(conversation) < 3:
        stats["skipped_too_short_or_empty"] += 1
        continue

    # Total character length check
    full_text_length = sum(len((m.get("value") or "")) for m in conversation)
    if full_text_length > MAX_TOTAL_CONV_LENGTH:
        stats["skipped_total_too_long"] += 1
        continue

    # Check GPT completion lengths
    if any(
        m.get("from") == "gpt" and len((m.get("value") or "")) > MAX_COMPLETION_LENGTH
        for m in conversation
    ):
        stats["skipped_completion_too_long"] += 1
        continue

    stats["kept_conversations"] += 1

    # Extract GPT -> human interaction pairs
    for i in range(len(conversation) - 1):
        gpt_msg = conversation[i]
        human_reply = conversation[i + 1]

        if gpt_msg.get("from") != "gpt":
            continue
        if human_reply.get("from") != "human":
            continue

        history_full = conversation[:i]

        prompt_history = truncate_history_starting_with_human(
            history_full,
            MAX_HISTORY_MESSAGES
        )

        if prompt_history is None:
            stats["skipped_no_valid_history"] += 1
            continue

        entry = {
            "id": f"{original_idx}_{i}",
            "original_conv_id": original_idx,
            "turn_id": i,
            "prompt": prompt_history,
            "completion": gpt_msg,
            "user_response": human_reply,
            "prompt_len": len(prompt_history),
            "completion_len": len((gpt_msg.get("value") or "")),
            "total_conv_len": full_text_length,
        }

        processed_data.append(entry)

df = pd.DataFrame(processed_data)

print("\n=== Processing Statistics ===")
print(f"Total source conversations:      {stats['total_conversations']}")
print(f"Skipped (total too long):        {stats['skipped_total_too_long']}")
print(f"Skipped (completion too long):   {stats['skipped_completion_too_long']}")
print(f"Skipped (too short/empty):       {stats['skipped_too_short_or_empty']}")
print(f"Skipped (no valid history):      {stats['skipped_no_valid_history']}")
print(f"Conversations kept:              {stats['kept_conversations']}")
print(f"Interaction pairs generated:     {len(df)}")
print("================================\n")

data_dir = (Path.cwd().parent / "data").resolve()
data_dir.mkdir(parents=True, exist_ok=True)

output_path = data_dir / OUTPUT_FILENAME

if not df.empty:
    df.to_json(output_path, orient="records", lines=True)
    print(f"Saved filtered data to: {output_path}")
else:
    print("Warning: No data remained after filtering!")
