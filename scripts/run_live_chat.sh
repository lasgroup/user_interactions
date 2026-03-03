#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
LR="${LR:-1e-5}"
LORA_R="${LORA_R:-64}"
SIGNAL_CLIP="${SIGNAL_CLIP:-2.0}"
PORT="${PORT:-7860}"
CKPT_DIR="${CKPT_DIR:-./live_checkpoints}"
CKPT_EVERY="${CKPT_EVERY:-10}"

python "$REPO_ROOT/live_chat.py" \
  --model "$MODEL" \
  --lr "$LR" \
  --lora_r "$LORA_R" \
  --signal_clip "$SIGNAL_CLIP" \
  --port "$PORT" \
  --checkpoint_dir "$CKPT_DIR" \
  --checkpoint_every "$CKPT_EVERY" \
  "$@"
