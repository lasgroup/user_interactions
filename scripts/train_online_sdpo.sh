#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/train_online_sdpo.sh [--dry-run]
#
# Common overrides:
#   MODEL_NAME_OR_PATH="Qwen/Qwen3-8B" ./scripts/train_online_sdpo.sh
#   TRAIN_JSONL=/data/train.jsonl VAL_JSONL=/data/val.jsonl ./scripts/train_online_sdpo.sh
#   WORLD_SIZE=4 ./scripts/train_online_sdpo.sh
#   ACCELERATE_CONFIG=./configs/accelerate_multigpu.yaml ./scripts/train_online_sdpo.sh

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
# Run configuration (your defaults; override via env vars)
# =============================================================================
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}"
LR="${LR:-5e-6}"
BS="${BS:-2}"
GA="${GA:-4}"
STYLE="${STYLE:-no_emojis}"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-general}"  # tldr|general

TRAIN_JSONL="${TRAIN_JSONL:-/path/to/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/path/to/validation.jsonl}"
TRAIN_N="${TRAIN_N:-512}"
EVAL_N="${EVAL_N:-256}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}"
SEED="${SEED:-42}"

# =============================================================================
# Output + caches (portable)
# =============================================================================
BASE_WORK="${BASE_WORK:-${SCRATCH:-${TMPDIR:-/tmp}}}"
RUN_ID="${RUN_ID:-sdpo-$(date +%Y%m%d-%H%M%S)}"

OUTPUT_DIR="${OUTPUT_DIR:-$BASE_WORK/sdpo-runs/$RUN_ID}"
CACHE_DIR="${CACHE_DIR:-$BASE_WORK/sdpo-cache/$RUN_ID}"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"/{hf,datasets,hub,wandb,pip}

export OUTPUT_DIR
export HF_HOME="$CACHE_DIR/hf"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/hub"
export WANDB_DIR="$CACHE_DIR/wandb"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Tracking (safe defaults; user can disable wandb via WANDB_MODE=disabled)
export WANDB_PROJECT="${WANDB_PROJECT:-helpsteer}"
export WANDB_NAME="${WANDB_NAME:-sdpo-${STYLE}-lr${LR}-bs${BS}-ga${GA}-${RUN_ID}}"

#   export HF_TOKEN=...
unset SSL_CERT_FILE SSL_CERT_DIR || true

# =============================================================================
# Command
# =============================================================================
cd "$REPO_ROOT"

SCRIPT_ARGS="\"$TRAIN_SCRIPT\" \
  --learning_rate \"$LR\" \
  --per_device_train_batch_size \"$BS\" \
  --gradient_accumulation_steps \"$GA\" \
  --style \"$STYLE\" \
  --model_name_or_path \"$MODEL_NAME_OR_PATH\" \
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
echo "LR=$LR BS=$BS GA=$GA STYLE=$STYLE SYSTEM_PROMPT=$SYSTEM_PROMPT"
echo "TRAIN_JSONL=$TRAIN_JSONL"
echo "VAL_JSONL=$VAL_JSONL"
echo

if [[ "${WORLD_SIZE}" -le 1 ]]; then
  run "python $SCRIPT_ARGS"
else
  if [[ -n "$ACCELERATE_CONFIG" ]]; then
    run "accelerate launch --config_file \"$ACCELERATE_CONFIG\" $SCRIPT_ARGS"
  else
    run "accelerate launch --num_processes \"$WORLD_SIZE\" $SCRIPT_ARGS"
  fi
fi
