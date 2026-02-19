#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/train_offline_sdpo.sh [--dry-run]
#
# Common overrides:
#   BASE_MODEL="Qwen/Qwen3-8B" ./scripts/train_offline_sdpo.sh
#   LR=2e-6 BS=4 GA=8 ./scripts/train_offline_sdpo.sh
#   WORLD_SIZE=4 ./scripts/train_offline_sdpo.sh
#   ACCELERATE_CONFIG=./configs/accelerate_multigpu.yaml ./scripts/train_offline_sdpo.sh

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
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/main_offline_sdpo.py}"

# Optional accelerate config. If unset, we do `accelerate launch --num_processes $WORLD_SIZE`
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
WORLD_SIZE="${WORLD_SIZE:-4}"

# =============================================================================
# Run configuration (defaults reflect your sbatch)
# =============================================================================
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
LR="${LR:-1e-6}"
BS="${BS:-4}"
GA="${GA:-8}"

# Tracking (safe defaults)
WANDB_PROJECT="${WANDB_PROJECT:-wildchat}"
WANDB_NAME="${WANDB_NAME:-offline-sdpo-${BASE_MODEL//\//-}-lr${LR}-bs${BS}-ga${GA}}"

# =============================================================================
# Output + caches (portable)
# =============================================================================
BASE_WORK="${BASE_WORK:-${SCRATCH:-${TMPDIR:-/tmp}}}"
RUN_ID="${RUN_ID:-sdpo-offline-$(date +%Y%m%d-%H%M%S)}"

OUTPUT_DIR="${OUTPUT_DIR:-$BASE_WORK/sdpo-offline-runs/$RUN_ID}"
CACHE_DIR="${CACHE_DIR:-$BASE_WORK/sdpo-offline-cache/$RUN_ID}"

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

PY_CMD="python \"$TRAIN_SCRIPT\" \
  --learning_rate \"$LR\" \
  --batch_size \"$BS\" \
  --grad_accum \"$GA\" \
  --base_model \"$BASE_MODEL\""

echo "REPO_ROOT=$REPO_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CACHE_DIR=$CACHE_DIR"
echo "BASE_MODEL=$BASE_MODEL"
echo "LR=$LR BS=$BS GA=$GA"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_NAME=$WANDB_NAME"
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
