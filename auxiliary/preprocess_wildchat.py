# preprocess_wildchat.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset


# --- Configuration ---
DATASET_NAME = "allenai/WildChat"
SPLIT = "train"

# Filtering
MIN_MESSAGES = 3  # after removing empty/invalid role messages
ENGLISH_VALUES = {"english", "en"}  # WildChat viewer commonly shows "English"

# Length constraints (NOTE: these are CHAR limits, not token limits)
MAX_TOTAL_CONV_LENGTH = 100_000   # drop conversation if sum(chars) exceeds this
MAX_COMPLETION_LENGTH = 4096      # drop conversation if ANY assistant msg exceeds this
MAX_USER_MESSAGE_LENGTH = 4096    # drop conversation if ANY user msg exceeds this

# History truncation
MAX_HISTORY_MESSAGES = 5          # max messages kept in prompt history

# Sampling (reproducible)
SEED = 42
TARGET_NUM_CONVERSATIONS = 15_000  # "about 14–15k"

# Optional: try to hit ~50k interactions without wildly overshooting conv count
TARGET_NUM_INTERACTIONS = 50_000
MIN_CONVERSATIONS_BEFORE_EARLY_STOP = 14_000  # don't stop too early on weird distributions

# Output
OUTPUT_FILENAME = "wildchat_interactions_v1.jsonl"

# Extraction mode:
# - "assistant_to_user": records where completion is assistant msg and user_response is the NEXT user msg (like your WildFeedback logic)
# - "user_to_assistant": records where completion is assistant msg that answers a user msg; user_response is OPTIONAL next user msg
PAIR_KIND = "assistant_to_user"  # keep this to match your LRAS setup


def is_english(conversation_level_language: Optional[str]) -> bool:
    if not conversation_level_language:
        return False
    return conversation_level_language.strip().lower() in ENGLISH_VALUES


def normalize_conversation(raw_conv: List[Dict]) -> List[Dict[str, str]]:
    """
    Convert WildChat messages (role/content) into your old format (from/value),
    dropping empty messages and unknown roles.
    """
    out: List[Dict[str, str]] = []
    for m in raw_conv:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            out.append({"from": "human", "value": content})
        elif role == "assistant":
            out.append({"from": "gpt", "value": content})
        else:
            # ignore anything unexpected
            continue
    return out


def truncate_history_starting_with_human(
    history: List[Dict[str, str]],
    max_messages: int
) -> Optional[List[Dict[str, str]]]:
    """
    Truncate history to at most `max_messages` messages such that:
    - starts with a human message
    - alternates roles
    - no empty messages (assumed already removed)
    Returns None if no valid history remains.
    """
    if not history:
        return None

    truncated = history[-max_messages:]

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

    if not normalized or normalized[0]["from"] != "human":
        return None

    return normalized


def should_stop(kept_convs: int, interactions: int) -> bool:
    if kept_convs >= TARGET_NUM_CONVERSATIONS:
        return True

    if TARGET_NUM_INTERACTIONS is not None:
        if kept_convs >= MIN_CONVERSATIONS_BEFORE_EARLY_STOP and interactions >= TARGET_NUM_INTERACTIONS:
            return True

    return False


def main() -> None:
    print(f"Loading dataset {DATASET_NAME} [{SPLIT}] ...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    # Reproducible sampling order
    ds = ds.shuffle(seed=SEED)

    data_dir = (Path.cwd().parent / "data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / OUTPUT_FILENAME

    stats = {
        "total_conversations_scanned": 0,
        "kept_conversations": 0,
        "kept_interactions": 0,
        "skipped_non_english": 0,
        "skipped_too_short_or_empty": 0,
        "skipped_total_too_long": 0,
        "skipped_completion_too_long": 0,
        "skipped_user_too_long": 0,
        "skipped_no_valid_history": 0,
    }

    print(
        f"Processing...\n"
        f"  PAIR_KIND:                {PAIR_KIND}\n"
        f"  Target conversations:     {TARGET_NUM_CONVERSATIONS}\n"
        f"  Target interactions:      {TARGET_NUM_INTERACTIONS}\n"
        f"  Seed:                     {SEED}\n"
        f"  Min messages:             {MIN_MESSAGES}\n"
        f"  Max total conv length:    {MAX_TOTAL_CONV_LENGTH}\n"
        f"  Max assistant msg length: {MAX_COMPLETION_LENGTH}\n"
        f"  Max user msg length:      {MAX_USER_MESSAGE_LENGTH}\n"
        f"  Max history messages:     {MAX_HISTORY_MESSAGES}\n"
        f"  Output:                   {output_path}\n"
    )

    with output_path.open("w", encoding="utf-8") as f:
        for row_idx, row in enumerate(ds):
            stats["total_conversations_scanned"] += 1

            # Language filter (conversation-level)
            if not is_english(row.get("language")):
                stats["skipped_non_english"] += 1
                continue

            raw_conv = row.get("conversation") or []
            conv_id = row.get("conversation_id")
            model = row.get("model")
            timestamp = row.get("timestamp")

            norm_conv = normalize_conversation(raw_conv)

            if len(norm_conv) < MIN_MESSAGES:
                stats["skipped_too_short_or_empty"] += 1
                continue

            # Total char length check
            full_text_length = sum(len(m["value"]) for m in norm_conv)
            if full_text_length > MAX_TOTAL_CONV_LENGTH:
                stats["skipped_total_too_long"] += 1
                continue

            # Any assistant message too long?
            if any(m["from"] == "gpt" and len(m["value"]) > MAX_COMPLETION_LENGTH for m in norm_conv):
                stats["skipped_completion_too_long"] += 1
                continue

            # Any user message too long?
            if any(m["from"] == "human" and len(m["value"]) > MAX_USER_MESSAGE_LENGTH for m in norm_conv):
                stats["skipped_user_too_long"] += 1
                continue

            stats["kept_conversations"] += 1

            # Extract interactions
            if PAIR_KIND == "assistant_to_user":
                # completion = assistant at i, user_response = next user at i+1
                for i in range(len(norm_conv) - 1):
                    completion = norm_conv[i]
                    user_response = norm_conv[i + 1]

                    if completion["from"] != "gpt":
                        continue
                    if user_response["from"] != "human":
                        continue

                    history_full = norm_conv[:i]
                    prompt_history = truncate_history_starting_with_human(
                        history_full,
                        MAX_HISTORY_MESSAGES
                    )

                    # Must exist AND end with a human message (best for add_generation_prompt=True)
                    if prompt_history is None or prompt_history[-1]["from"] != "human":
                        stats["skipped_no_valid_history"] += 1
                        continue

                    entry = {
                        "id": f"{conv_id}_{i}" if conv_id else f"{row_idx}_{i}",
                        "conversation_id": conv_id,
                        "row_idx": row_idx,
                        "turn_id": i,
                        "model": model,
                        "timestamp": str(timestamp) if timestamp is not None else None,
                        "prompt": prompt_history,
                        "completion": completion,
                        "user_response": user_response,
                        "prompt_len": len(prompt_history),
                        "completion_len": len(completion["value"]),
                        "user_response_len": len(user_response["value"]),
                        "total_conv_len": full_text_length,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    stats["kept_interactions"] += 1

            elif PAIR_KIND == "user_to_assistant":
                # completion = assistant at i+1 answering user at i
                for i in range(len(norm_conv) - 1):
                    user_msg = norm_conv[i]
                    completion = norm_conv[i + 1]
                    if user_msg["from"] != "human":
                        continue
                    if completion["from"] != "gpt":
                        continue

                    # history includes the user message that the assistant answers
                    history_full = norm_conv[: i + 1]
                    prompt_history = truncate_history_starting_with_human(
                        history_full,
                        MAX_HISTORY_MESSAGES
                    )

                    if prompt_history is None or prompt_history[-1]["from"] != "human":
                        stats["skipped_no_valid_history"] += 1
                        continue

                    # optional next user reply
                    user_response = None
                    if i + 2 < len(norm_conv) and norm_conv[i + 2]["from"] == "human":
                        user_response = norm_conv[i + 2]

                    entry = {
                        "id": f"{conv_id}_{i+1}" if conv_id else f"{row_idx}_{i+1}",
                        "conversation_id": conv_id,
                        "row_idx": row_idx,
                        "turn_id": i + 1,
                        "model": model,
                        "timestamp": str(timestamp) if timestamp is not None else None,
                        "prompt": prompt_history,
                        "completion": completion,
                        "user_response": user_response,
                        "prompt_len": len(prompt_history),
                        "completion_len": len(completion["value"]),
                        "user_response_len": len(user_response["value"]) if user_response else None,
                        "total_conv_len": full_text_length,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    stats["kept_interactions"] += 1

            else:
                raise ValueError(f"Unknown PAIR_KIND={PAIR_KIND!r}")

            if should_stop(stats["kept_conversations"], stats["kept_interactions"]):
                break

    print("\n=== Processing Statistics ===")
    for k, v in stats.items():
        print(f"{k:30s} {v}")
    print("============================\n")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
