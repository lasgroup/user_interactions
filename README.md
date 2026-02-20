# User Interactions — SDPO Training Code

This repository contains the training code for **Self-Distillation Policy Optimization (SDPO) from User Interactions**.

The core idea: at each step the policy generates a response `y` to a prompt `x`, a user simulator produces a follow-up `o`, and the per-token log-ratio `log p(y | x, o) − log p(y | x)` is used as a token-level advantage signal to update the policy.

The repo includes an online trainer build upon a policy gradient implementation from TRL and an offline version SDPO. 

- **Continual Personalization and Adaptation (Online SDPO)** — the policy generates responses on-the-fly; signal is computed immediately against the current model.
  - *TL;DR summarization* (`openai/summarize_from_feedback`) with a local Qwen3-8B user simulator
  - *General prompts* (`nvidia/HelpSteer2`) with a Claude Haiku user simulator
- **General Alignment from Logged Real-World Conversations (Offline SDPO)** — signal is computed from existing interaction data from raw conversations from *WildFeedback* (`microsoft/WildFeedback`) or *WildChat* (`allenai/WildChat`)

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch==2.7.0`, `transformers==4.57.6`, `accelerate==1.6.0`, `trl==0.24.0`, `datasets==4.5.0`, `peft==0.15.1`, `bitsandbytes`, `wandb`, `anthropic`.

Set your credentials before running:

```bash
export WANDB_API_KEY=...
export HF_TOKEN=...           # if model downloads require authentication
export ANTHROPIC_API_KEY=...  # needed for Claude user simulator and Claude judge
```

---

## Online Experiment 1 — TL;DR summarization (local user simulator)

**Dataset:** `openai/summarize_from_feedback`
**User simulator:** local Qwen3-4B (`StyleUserSimulator`, no API key needed)
**Default style:** `concise_casual_beginner`

### 1. Prepare data (once)

```bash
./scripts/prepare_data_tldr.sh
# Writes to ./data/tldr_prompts_unique/train.jsonl and validation.jsonl
```

### 2. Train

```bash
TRAIN_JSONL=./data/tldr_prompts_unique/train.jsonl \
VAL_JSONL=./data/tldr_prompts_unique/validation.jsonl \
./scripts/train_online_sdpo_tldr.sh
```

Single GPU:

```bash
WORLD_SIZE=1 \
TRAIN_JSONL=./data/tldr_prompts_unique/train.jsonl \
VAL_JSONL=./data/tldr_prompts_unique/validation.jsonl \
./scripts/train_online_sdpo_tldr.sh
```

**Common overrides:**

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen3-4B` | Policy model |
| `USER_MODEL_NAME_OR_PATH` | `Qwen/Qwen3-4B` | Local user simulator model |
| `STYLE` | `concise_casual_beginner` | Target user style |
| `LR` | `5e-6` | Learning rate |
| `BS` | `4` | Per-device train batch size |
| `GA` | `1` | Gradient accumulation steps |
| `WORLD_SIZE` | `4` | Number of GPUs |
| `OUTPUT_DIR` | `$SCRATCH/sdpo-tldr-runs/<run-id>` | Checkpoint output directory |

### 3. Evaluate checkpoints

```bash
RUN_DIR=/path/to/sdpo-tldr-runs/run-id \
DATA_PATH=./data/tldr_prompts_unique/validation.jsonl \
./scripts/eval_checkpoints_tldr.sh
```

Compares each checkpoint against the base model using a local Qwen3 judge. Outputs a summary table of win-rates across checkpoints.

**Common overrides:**

| Variable | Default | Description |
|---|---|---|
| `CKPTS` | `3 6 9 12 15` | Checkpoint steps to evaluate |
| `BASELINE` | `Qwen/Qwen3-8B` | Model to compare against |
| `JUDGE_MODEL` | `Qwen/Qwen3-8B` | Local judge model |
| `EVAL_N` | `256` | Number of evaluation examples |
| `WORLD_SIZE` | `4` | Number of GPUs |

---

## Online Experiment 2 — General prompts (Claude Haiku user simulator)

**Dataset:** `nvidia/HelpSteer2`
**User simulator:** Claude Haiku 4.5 (`ClaudeStyleUserSimulator`, requires `ANTHROPIC_API_KEY`)
**Default style:** `no_emojis`

### 1. Prepare data (once)

```bash
./scripts/prepare_data_helpsteer.sh
# Writes to ./data/helpsteer2_prompts/train.jsonl and validation.jsonl
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
| `STYLE` | `no_emojis` | Target user style |
| `LR` | `5e-6` | Learning rate |
| `BS` | `2` | Per-device train batch size |
| `GA` | `4` | Gradient accumulation steps |
| `WORLD_SIZE` | `4` | Number of GPUs |
| `OUTPUT_DIR` | `$SCRATCH/sdpo-runs/<run-id>` | Checkpoint output directory |

### 3. Evaluate checkpoints

```bash
RUN_DIR=/path/to/sdpo-runs/run-id \
DATA_PATH=./data/helpsteer2_prompts/validation.jsonl \
./scripts/eval_checkpoints.sh
```

Compares each checkpoint against the base model using Claude Haiku as judge (requires `ANTHROPIC_API_KEY`).

**Common overrides:**

| Variable | Default | Description |
|---|---|---|
| `CKPTS` | `3 6 9 12 15` | Checkpoint steps to evaluate |
| `BASELINE` | `Qwen/Qwen3-8B` | Model to compare against |
| `JUDGE_MODEL` | `claude-haiku-4-5-20251001` | Claude judge model |
| `EVAL_N` | `256` | Number of evaluation examples |
| `WORLD_SIZE` | `4` | Number of GPUs |

---

## Offline Experiment — Training on WildFeedback / WildChat

In the offline setting the signal is computed from real interaction data rather than a live simulator.

### WildFeedback (`microsoft/WildFeedback`)

**1. Prepare data (once)**

```bash
./scripts/prepare_data_wildfeedback.sh
# Writes to ./data/wildfeedback/wildfeedback_interactions.jsonl
```

**2. Train**

```bash
TRAIN_JSONL=./data/wildfeedback/wildfeedback_interactions.jsonl \
./scripts/train_offline_sdpo.sh
```

### WildChat (`allenai/WildChat`)

**1. Prepare data (once)**

```bash
./scripts/prepare_data_wildchat.sh
# Writes to ./data/wildchat/wildchat_interactions_v1.jsonl
```

**2. Train**

```bash
TRAIN_JSONL=./data/wildchat/wildchat_interactions_v1.jsonl \
./scripts/train_offline_sdpo.sh
```

**Common overrides for offline training:**

| Variable | Default | Description |
|---|---|---|
| `BASE_MODEL` | `Qwen/Qwen3-8B` | Policy model |
| `LR` | `2e-6` | Learning rate |
| `BS` | `4` | Per-device train batch size |
| `GA` | `8` | Gradient accumulation steps |
| `WORLD_SIZE` | `4` | Number of GPUs |
| `OUTPUT_DIR` | `$SCRATCH/sdpo-offline-runs/<run-id>` | Checkpoint output directory |

---

## Signal visualization

`sdpo_signal_analysis.py` computes and visualizes the per-token SDPO signal for a set of prompt/feedback cases. It generates per-token heatmaps comparing the signal under an unrelated follow-up (should be near zero) and a relevant follow-up (should have structure).

```bash
./scripts/run_signal_analysis.sh
# Outputs to $SCRATCH/signal-analysis/<run-id>/
```

**Common overrides:**

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-8B` | Model to score with |
| `CASES_JSON` | `auxiliary/signal_analysis_cases.json` | Input cases |
| `N_CASES` | `24` | Number of cases to run |
| `OUT_DIR` | `$SCRATCH/signal-analysis/<run-id>` | Output directory |

Outputs: `sdpo_signals.json`, `unrelated.png`, `followup.png`, `stacked.png`, `side_by_side.png`, `case{N}_tokens.png`.

---

## Dry-run mode

All scripts accept `--dry-run`, which prints the resolved command without executing it:

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

