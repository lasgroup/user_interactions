# User Interactions — SDPO Training Code

This repository contains the training code for **Self-Distillation Policy Optimization (SDPO) from User Interactions**. This code covers the **online experiments** from the paper, which run on two tasks:

- **TL;DR summarization** — prompts from `openai/summarize_from_feedback`, user simulator is a local Qwen3-8B model
- **General prompts** — prompts from `nvidia/HelpSteer2`, user simulator is Claude Haiku

At each training step the policy generates a response, a user simulator reacts to it, and the log-ratio `log p(y | x, o) − log p(y | x)` (where `o` is the simulated user response) is used as a per-token advantage signal to update the policy.

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `trl==0.24.0`, `transformers`, `accelerate`, `bitsandbytes`, `wandb`, `anthropic`.

Set your credentials before training:

```bash
export WANDB_API_KEY=...
export HF_TOKEN=...          # if Qwen3-8B might require authentication on your account
export ANTHROPIC_API_KEY=... # Needed only if you want to use a Claude model to simulate user responses and evaluate the final policy performance
```

---

## Experiment 1 — TL;DR summarization with a local user simulator

**Dataset:** `openai/summarize_from_feedback`
**User simulator:** local Qwen3-8B (`StyleUserSimulator`)
**Default style:** `concise_casual_beginner`

### 1. Prepare data (once)

```bash
./scripts/prepare_data_tldr.sh
# Writes to ./data/tldr_prompts_unique/train.jsonl and validation.jsonl
```

Override the output directory:

```bash
OUT_DIR=/path/to/tldr_data ./scripts/prepare_data_tldr.sh
```

### 2. Train

```bash
TRAIN_JSONL=./data/tldr_prompts_unique/train.jsonl \
VAL_JSONL=./data/tldr_prompts_unique/validation.jsonl \
./scripts/train_online_sdpo_tldr.sh
```

The script defaults to `WORLD_SIZE=4` (multi-GPU via `accelerate launch`). For a single GPU:

```bash
WORLD_SIZE=1 \
TRAIN_JSONL=./data/tldr_prompts_unique/train.jsonl \
VAL_JSONL=./data/tldr_prompts_unique/validation.jsonl \
./scripts/train_online_sdpo_tldr.sh
```

**Common overrides:**

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen3-8B` | Policy model |
| `USER_MODEL_NAME_OR_PATH` | `Qwen/Qwen3-8B` | Local user simulator model |
| `STYLE` | `concise_casual_beginner` | Target user style (see `user_simulator.py` for all styles) |
| `LR` | `5e-6` | Learning rate |
| `BS` | `2` | Per-device train batch size |
| `GA` | `4` | Gradient accumulation steps |
| `WORLD_SIZE` | `4` | Number of GPUs |
| `OUTPUT_DIR` | `$SCRATCH/sdpo-tldr-runs/<run-id>` | Checkpoint output directory |

---

## Experiment 2 — Helpsteer2 with Claude Haiku user simulator

**Dataset:** `nvidia/HelpSteer2`
**User simulator:** Claude Haiku 4.5 (`ClaudeStyleUserSimulator`, requires `ANTHROPIC_API_KEY`)
**Default style:** `concise_casual_beginner`

### 1. Prepare data (once)

```bash
./scripts/prepare_data_helpsteer.sh
# Writes to ./data/helpsteer2_prompts/train.jsonl and validation.jsonl
```

Override the output directory:

```bash
OUT_DIR=/path/to/helpsteer_data ./scripts/prepare_data_helpsteer.sh
```

### 2. Train

```bash
TRAIN_JSONL=./data/helpsteer2_prompts/train.jsonl \
VAL_JSONL=./data/helpsteer2_prompts/validation.jsonl \
./scripts/train_online_sdpo.sh
```

**Common overrides:**

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen3-8B` | Policy model |
| `STYLE` | `concise_casual_beginner` | Target user style |
| `LR` | `5e-6` | Learning rate |
| `BS` | `4` | Per-device train batch size |
| `GA` | `1` | Gradient accumulation steps |
| `WORLD_SIZE` | `4` | Number of GPUs |
| `OUTPUT_DIR` | `$SCRATCH/sdpo-runs/<run-id>` | Checkpoint output directory |

---

## Dry-run mode

All training and data preparation scripts accept `--dry-run`, which prints the resolved command without executing it — useful for verifying configuration before a long run:

```bash
TRAIN_JSONL=./data/tldr_prompts_unique/train.jsonl \
VAL_JSONL=./data/tldr_prompts_unique/validation.jsonl \
./scripts/train_online_sdpo_tldr.sh --dry-run
```

## Multi-GPU with a custom accelerate config

```bash
ACCELERATE_CONFIG=./multigpu_accelerate_config.yaml \
TRAIN_JSONL=... VAL_JSONL=... \
./scripts/train_online_sdpo_tldr.sh
```

## User styles

The full list of available style personas is defined in `user_simulator.py` under `STYLE_PERSONAS`. Examples: `concise_casual_beginner`, `detailed_professional_expert`, `concise_casual_expert`, `poetic`, and many more.
