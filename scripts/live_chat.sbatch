#!/bin/bash
#SBATCH --job-name=live-sdpo
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000
#SBATCH --time=04:00:00
#SBATCH --output=/users/%u/output/sdpo/%j.log
#SBATCH --error=/users/%u/output/sdpo/%j.err
#SBATCH --environment=sample-efficient-rlvr
#SBATCH --exclude=nid007005,nid007004,nid006606,nid006684,nid006638,nid006569,nid006622,nid006571,nid006877,nid006817,nid006866,nid006927,nid007223,nid007132,nid007645,nid007557,nid007563,nid007590,nid006988,nid007648,nid007454,nid007191,nid006652

set -euo pipefail

# Fix SSL issues on HPC nodes
unset SSL_CERT_FILE SSL_CERT_DIR

REPO_ROOT="${REPO_ROOT:-$HOME/user_interactions}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Ensure output dir exists
mkdir -p "$HOME/output/sdpo"

echo "=== LIVE SDPO CHAT ==="
echo "Node:  $(hostname)"
echo "GPUs:  $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Date:  $(date)"
echo ""

python "$REPO_ROOT/live_chat.py" \
  --model "${MODEL:-Qwen/Qwen3-8B}" \
  --lr "${LR:-1e-5}" \
  --share \
  "$@"
