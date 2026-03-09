#!/usr/bin/env bash
set -euo pipefail

# Supervised fine-tuning on WildFeedback (or any JSONL).
#
# Usage:
#   TRAIN_JSONL=/path/to/wildfeedback_interactions.jsonl ./scripts/sft_wildfeedback.sbatch [--dry-run]
#
# Common overrides:
#   BASE_MODEL="Qwen/Qwen3-8B" ./scripts/sft_wildfeedback.sbatch
#   LR=2e-6 BS=4 GA=2 ./scripts/sft_wildfeedback.sbatch
#   WORLD_SIZE=4 ./scripts/sft_wildfeedback.sbatch
#   ACCELERATE_CONFIG=./multigpu_accelerate_config.yaml ./scripts/sft_wildfeedback.sbatch

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
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/test-time-alignment/sft_wildfeedback.py}"

# Optional accelerate config. If unset, we do `accelerate launch --num_processes $WORLD_SIZE`
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
WORLD_SIZE="${WORLD_SIZE:-4}"

# =============================================================================
# Run configuration
# =============================================================================
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
LR="${LR:-2e-6}"
BS="${BS:-4}"
GA="${GA:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"

TRAIN_JSONL="${TRAIN_JSONL:-}"  # REQUIRED (e.g. /path/to/wildfeedback_interactions.jsonl)

if [[ -z "$TRAIN_JSONL" ]]; then
  echo "ERROR: TRAIN_JSONL is required (e.g. /path/to/wildfeedback_interactions.jsonl)"
  exit 1
fi

# Tracking
WANDB_PROJECT="${WANDB_PROJECT:-wildfeedback-sft}"
WANDB_NAME="${WANDB_NAME:-sft-${BASE_MODEL//\//-}-lr${LR}-bs${BS}-ga${GA}}"

# =============================================================================
# Output + caches (portable)
# =============================================================================
BASE_WORK="${BASE_WORK:-${SCRATCH:-${TMPDIR:-/tmp}}}"
RUN_ID="${RUN_ID:-sft-$(date +%Y%m%d-%H%M%S)}"

OUTPUT_DIR="${OUTPUT_DIR:-$BASE_WORK/sft-runs/$RUN_ID}"
CACHE_DIR="${CACHE_DIR:-$BASE_WORK/sft-cache/$RUN_ID}"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"/{hf,datasets,hub,wandb,pip}

export OUTPUT_DIR
export HF_HOME="$CACHE_DIR/hf"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/hub"
export WANDB_DIR="$CACHE_DIR/wandb"
export PIP_CACHE_DIR="$CACHE_DIR/pip"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export WANDB_PROJECT
export WANDB_NAME

#   export HF_TOKEN=...
unset SSL_CERT_FILE SSL_CERT_DIR || true

# =============================================================================
# Command
# =============================================================================
cd "$REPO_ROOT"

SCRIPT_ARGS="\"$TRAIN_SCRIPT\" \
  --model \"$BASE_MODEL\" \
  --data \"$TRAIN_JSONL\" \
  --out \"$OUTPUT_DIR\" \
  --max_length \"$MAX_LENGTH\" \
  --epochs \"$NUM_EPOCHS\" \
  --lr \"$LR\" \
  --bsz \"$BS\" \
  --gas \"$GA\""

echo "REPO_ROOT=$REPO_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CACHE_DIR=$CACHE_DIR"
echo "BASE_MODEL=$BASE_MODEL"
echo "LR=$LR BS=$BS GA=$GA"
echo "MAX_LENGTH=$MAX_LENGTH NUM_EPOCHS=$NUM_EPOCHS"
echo "TRAIN_JSONL=$TRAIN_JSONL"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_NAME=$WANDB_NAME"
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
