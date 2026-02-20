#!/usr/bin/env bash
set -euo pipefail

# Online SDPO training on TL;DR summarization prompts (openai/summarize_from_feedback).
#
# Usage:
#   ./scripts/train_online_sdpo_tldr.sh [--dry-run]
#
# Common overrides:
#   MODEL_NAME_OR_PATH="Qwen/Qwen3-4B" ./scripts/train_online_sdpo_tldr.sh
#   TRAIN_JSONL=/data/tldr/train.jsonl VAL_JSONL=/data/tldr/validation.jsonl ./scripts/train_online_sdpo_tldr.sh
#   WORLD_SIZE=4 ./scripts/train_online_sdpo_tldr.sh
#   ACCELERATE_CONFIG=./configs/accelerate_multigpu.yaml ./scripts/train_online_sdpo_tldr.sh
#
# Data preparation (run once before training):
#   python auxiliary/process_tldr_dataset.py --out_dir /path/to/tldr_data

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "Dry run mode enabled. Commands will be printed but not executed."
fi

run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "$*"
  else
    eval "$*"
  fi
}

# =============================================================================
# Paths
# =============================================================================
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/main_online_sdpo.py}"

# Optional accelerate config. If unset, we do `accelerate launch --num_processes $WORLD_SIZE`
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
WORLD_SIZE="${WORLD_SIZE:-4}"

# =============================================================================
# Run configuration
# =============================================================================
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
USER_MODEL_NAME_OR_PATH="${USER_MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
LR="${LR:-5e-6}"
BS="${BS:-4}"
GA="${GA:-1}"
STYLE="${STYLE:-concise_casual_beginner}"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-tldr}"  # tldr|general (matches your python CLI)

TRAIN_JSONL="${TRAIN_JSONL:-/path/to/tldr_data/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/path/to/tldr_data/validation.jsonl}"
TRAIN_N="${TRAIN_N:-512}"
EVAL_N="${EVAL_N:-256}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}"
SEED="${SEED:-43}"

# =============================================================================
# Output + caches (portable)
# =============================================================================
BASE_WORK="${BASE_WORK:-${SCRATCH:-${TMPDIR:-/tmp}}}"
RUN_ID="${RUN_ID:-sdpo-tldr-$(date +%Y%m%d-%H%M%S)}"

OUTPUT_DIR="${OUTPUT_DIR:-$BASE_WORK/sdpo-tldr-runs/$RUN_ID}"
CACHE_DIR="${CACHE_DIR:-$BASE_WORK/sdpo-tldr-cache/$RUN_ID}"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"/{hf,datasets,hub,wandb,pip}

export OUTPUT_DIR
export HF_HOME="$CACHE_DIR/hf"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/hub"
export WANDB_DIR="$CACHE_DIR/wandb"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export WANDB_PROJECT="${WANDB_PROJECT:-tldr}"
export WANDB_NAME="${WANDB_NAME:-sdpo-tldr-${STYLE}-lr${LR}-bs${BS}-ga${GA}-${RUN_ID}}"

#   export HF_TOKEN=...
#   export ANTHROPIC_API_KEY=...
unset SSL_CERT_FILE SSL_CERT_DIR || true

# =============================================================================
# Command
# =============================================================================
cd "$REPO_ROOT"

PY_CMD="python \"$TRAIN_SCRIPT\" \
  --learning_rate \"$LR\" \
  --per_device_train_batch_size \"$BS\" \
  --gradient_accumulation_steps \"$GA\" \
  --style \"$STYLE\" \
  --model_name_or_path \"$MODEL_NAME_OR_PATH\" \
  --user_model_name_or_path \"$USER_MODEL_NAME_OR_PATH\" \
  --system_prompt \"$SYSTEM_PROMPT\" \
  --train_jsonl \"$TRAIN_JSONL\" \
  --val_jsonl \"$VAL_JSONL\" \
  --train_n \"$TRAIN_N\" \
  --eval_n \"$EVAL_N\" \
  --max_prompt_tokens \"$MAX_PROMPT_TOKENS\" \
  --seed \"$SEED\""

echo "REPO_ROOT=$REPO_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CACHE_DIR=$CACHE_DIR"
echo "MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH"
echo "LR=$LR BS=$BS GA=$GA STYLE=$STYLE SYSTEM_PROMPT=$SYSTEM_PROMPT USER_MODEL=$USER_MODEL_NAME_OR_PATH"
echo "TRAIN_JSONL=$TRAIN_JSONL"
echo "VAL_JSONL=$VAL_JSONL"
echo

if [[ "${WORLD_SIZE}" -le 1 ]]; then
  run "$PY_CMD"
else
  if [[ -n "$ACCELERATE_CONFIG" ]]; then
    run "accelerate launch --config_file \"$ACCELERATE_CONFIG\" $PY_CMD"
  else
    run "accelerate launch --num_processes \"$WORLD_SIZE\" $PY_CMD"
  fi
fi
