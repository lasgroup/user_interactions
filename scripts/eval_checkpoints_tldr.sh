#!/usr/bin/env bash
set -euo pipefail

# Evaluate multiple training checkpoints for the TL;DR experiment.
# Compares each checkpoint against the base model using a local Qwen3-8B judge.
#
# Usage:
#   RUN_DIR=/path/to/sdpo-tldr-runs/run-id \
#   DATA_PATH=/path/to/tldr_data/validation.jsonl \
#   ./scripts/eval_checkpoints_tldr.sh [--dry-run]
#
# Common overrides:
#   CKPTS="3 6 9 12 15" STYLE="concise_casual_beginner" ./scripts/eval_checkpoints_tldr.sh
#   BASELINE="Qwen/Qwen3-8B" JUDGE_MODEL="Qwen/Qwen3-8B" ./scripts/eval_checkpoints_tldr.sh
#   WORLD_SIZE=4 ACCELERATE_CONFIG=./multigpu_accelerate_config.yaml ./scripts/eval_checkpoints_tldr.sh

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "Dry run mode enabled. Commands will be printed but not executed."
fi

run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "----------------------------------------------------------------"
    echo "$*"
  else
    echo "----------------------------------------------------------------"
    echo "$*"
    eval "$*"
  fi
}

# =============================================================================
# Paths
# =============================================================================
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_ROOT/auxiliary/eval_style_pairwise_accelerate.py}"

RUN_DIR="${RUN_DIR:-}"     # REQUIRED: directory containing checkpoint-* subdirs
DATA_PATH="${DATA_PATH:-}" # REQUIRED: path to validation.jsonl

if [[ -z "$RUN_DIR" ]]; then
  echo "ERROR: RUN_DIR is required and must point to a directory containing checkpoint-* subdirs"
  exit 1
fi
if [[ -z "$DATA_PATH" ]]; then
  echo "ERROR: DATA_PATH is required (e.g., /path/to/tldr_data/validation.jsonl)"
  exit 1
fi

# =============================================================================
# Eval configuration
# =============================================================================
STYLE="${STYLE:-concise_casual_beginner}"
BASELINE="${BASELINE:-Qwen/Qwen3-8B}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}"  # local model judge

# Must match the system prompt used during training
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Write summary of the text that is 1-2 sentences long. Always begin with 'TL;DR:' and output only the summary.}"

CKPTS="${CKPTS:-3 6 9 12 15}"
EVAL_N="${EVAL_N:-256}"
SEED="${SEED:-42}"
MAX_PROMPT_TOKENS_FILTER="${MAX_PROMPT_TOKENS_FILTER:-512}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-2048}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"

# =============================================================================
# Accelerate / compute
# =============================================================================
WORLD_SIZE="${WORLD_SIZE:-4}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"

# =============================================================================
# Output + caches
# =============================================================================
BASE_WORK="${BASE_WORK:-${SCRATCH:-${TMPDIR:-/tmp}}}"
RUN_ID="${RUN_ID:-eval-tldr-$(date +%Y%m%d-%H%M%S)}"

OUT_DIR="${OUT_DIR:-$BASE_WORK/sdpo-eval/$RUN_ID}"
CACHE_DIR="${CACHE_DIR:-$BASE_WORK/sdpo-eval-cache/$RUN_ID}"

mkdir -p "$OUT_DIR" "$CACHE_DIR"/{hf,datasets,hub,tmp}

export HF_HOME="$CACHE_DIR/hf"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/hub"
export TMPDIR="$CACHE_DIR/tmp"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

unset SSL_CERT_FILE SSL_CERT_DIR || true

cd "$REPO_ROOT"

echo "RUN_DIR:    $RUN_DIR"
echo "DATA_PATH:  $DATA_PATH"
echo "STYLE:      $STYLE"
echo "CKPTS:      $CKPTS"
echo "BASELINE:   $BASELINE"
echo "JUDGE:      $JUDGE_MODEL"
echo "OUT_DIR:    $OUT_DIR"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "---------------------------------------"

launch_eval() {
  local args="$1"
  if [[ "${WORLD_SIZE}" -le 1 ]]; then
    run "python $args"
  else
    if [[ -n "$ACCELERATE_CONFIG" ]]; then
      run "accelerate launch --config_file \"$ACCELERATE_CONFIG\" $args"
    else
      run "accelerate launch --num_processes \"$WORLD_SIZE\" $args"
    fi
  fi
}

declare -a SUMMARY_LINES=()

for CKPT in $CKPTS; do
  CANDIDATE="$RUN_DIR/checkpoint-$CKPT"
  RUN_NAME="tldr_${STYLE}_base_vs_ckpt${CKPT}"

  echo "[$(date '+%F %T')] Starting CKPT=$CKPT  CANDIDATE=$CANDIDATE"

  SCRIPT_ARGS="\"$EVAL_SCRIPT\" \
    --local_dataset_dir \"$DATA_PATH\" \
    --eval_split validation \
    --eval_n \"$EVAL_N\" \
    --seed \"$SEED\" \
    --max_prompt_tokens_filter \"$MAX_PROMPT_TOKENS_FILTER\" \
    --system_prompt \"$SYSTEM_PROMPT\" \
    --model_a_name_or_path \"$CANDIDATE\" \
    --model_b_name_or_path \"$BASELINE\" \
    --judge_model_name_or_path \"$JUDGE_MODEL\" \
    --style \"$STYLE\" \
    --max_input_tokens \"$MAX_INPUT_TOKENS\" \
    --max_new_tokens \"$MAX_NEW_TOKENS\" \
    --batch_size \"$BATCH_SIZE\" \
    --temperature \"$TEMPERATURE\" \
    --out_dir \"$OUT_DIR\" \
    --run_name \"$RUN_NAME\""

  launch_eval "$SCRIPT_ARGS"

  echo "[$(date '+%F %T')] Finished CKPT=$CKPT"

  OUT_JSON="$OUT_DIR/${RUN_NAME}.json"
  LINE="$(python - <<PY
import json, os
path = "$OUT_JSON"
ckpt = "$CKPT"
if not os.path.exists(path):
    print(f"{ckpt}\tMISSING\tMISSING\tMISSING\t0\t0\t0\t0\t0.0")
    raise SystemExit

rep = json.load(open(path))
m = rep["metrics"]

wins_a = int(m.get("wins_a", 0))
wins_b = int(m.get("wins_b", 0))
ties   = int(m.get("ties", 0))
n_eff  = int(m.get("n_effective_no_ties", wins_a + wins_b))

wr   = float(m.get("winrate_a_ignoring_ties", float("nan")))
se_a = float(m.get("winrate_a_ignoring_ties_se", float("nan")))
se_b = float(m.get("winrate_a_ignoring_ties_bootstrap_se", float("nan")))
coverage = float(m.get("coverage", 0.0))

print(f"{ckpt}\t{wr:.6f}\t{se_a:.6f}\t{se_b:.6f}\t{n_eff}\t{wins_a}\t{wins_b}\t{ties}\t{coverage:.6f}")
PY
)"
  SUMMARY_LINES+=("$LINE")

  echo "---------------------------------------"
done

echo "[$(date '+%F %T')] All checkpoints done."
echo ""
echo "STYLE:    $STYLE"
echo "BASELINE: $BASELINE"
echo "JUDGE:    $JUDGE_MODEL"
echo "================ TL;DR EVAL SUMMARY ================"
echo -e "ckpt\twinrate\tse_analytic\tse_boot\tN_eff\twins_a\twins_b\tties\tcoverage"
for L in "${SUMMARY_LINES[@]}"; do
  echo -e "$L"
done
echo "====================================================="
