#!/bin/bash
#SBATCH --job-name=eval-sdpo
#SBATCH --account=infra01
#SBATCH --reservation=PA-2338-RL
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --time=02:00:00
#SBATCH --output=/users/%u/output/sdpo/%j.log
#SBATCH --error=/users/%u/output/sdpo/%j.err
#SBATCH --environment=sample-efficient-rlvr
#SBATCH --exclude=nid007005,nid007004,nid006606,nid006684,nid006638,nid006569,nid006622,nid006571,nid006877,nid006817,nid006866,nid006927,nid007223,nid007132,nid007645,nid007557,nid007563,nid007590,nid006988,nid007648,nid007454,nid007191,nid006652

set -euo pipefail

# Fix SSL issues on HPC nodes
unset SSL_CERT_FILE SSL_CERT_DIR

REPO_ROOT="${REPO_ROOT:-$HOME/user_interactions}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/helpsteer_prompts}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Ensure output dirs exist
mkdir -p "$HOME/output/sdpo"

echo "=== EVAL LIVE SDPO ==="
echo "Node:  $(hostname)"
echo "GPUs:  $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Date:  $(date)"
echo ""

python "$REPO_ROOT/eval_live_sdpo.py" \
  --model "${MODEL:-Qwen/Qwen3-8B}" \
  --lr "${LR:-1e-5}" \
  --style "${STYLE:-writing_tics}" \
  --loss_mode "${LOSS_MODE:-full_distillation}" \
  --train_jsonl "$DATA_DIR/train.jsonl" \
  --val_jsonl "$DATA_DIR/validation.jsonl" \
  --train_n "${TRAIN_N:-50}" \
  --eval_n "${EVAL_N:-50}" \
  --eval_every "${EVAL_EVERY:-10}" \
  --use_lora \
  --use_vllm \
  --run_name "${RUN_NAME:-eval_live_sdpo}" \
  "$@"
