#!/usr/bin/env bash
set -euo pipefail

# Compute and visualize the SDPO per-token signal (log-ratio) on a set of
# prompt/feedback cases.
#
# Usage:
#   ./scripts/run_signal_analysis.sh [--dry-run]
#
# Common overrides:
#   MODEL="Qwen/Qwen3-8B" ./scripts/run_signal_analysis.sh
#   CASES_JSON=./auxiliary/signal_analysis_cases.json ./scripts/run_signal_analysis.sh
#   N_CASES=8 ./scripts/run_signal_analysis.sh
#
# Outputs in OUT_DIR:
#   sdpo_signals.json       — raw per-token signals for all cases
#   unrelated.png           — heatmap: P(y|x,o_unrelated) - P(y|x)
#   followup.png            — heatmap: P(y|x,o_followup)  - P(y|x)
#   stacked.png             — both heatmaps stacked vertically
#   side_by_side.png        — both heatmaps side by side
#   case{N}_tokens.png      — token-level colored boxes for one case

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
SCRIPT="${SCRIPT:-$REPO_ROOT/sdpo_signal_analysis.py}"

# =============================================================================
# Configuration
# =============================================================================
MODEL="${MODEL:-Qwen/Qwen3-8B}"
CASES_JSON="${CASES_JSON:-$REPO_ROOT/auxiliary/signal_analysis_cases.json}"
N_CASES="${N_CASES:-24}"
SEED="${SEED:-123}"

# =============================================================================
# Output + caches (portable)
# =============================================================================
BASE_WORK="${BASE_WORK:-${SCRATCH:-${TMPDIR:-/tmp}}}"
RUN_ID="${RUN_ID:-signal-analysis-$(date +%Y%m%d-%H%M%S)}"

OUT_DIR="${OUT_DIR:-$BASE_WORK/signal-analysis/$RUN_ID}"
CACHE_DIR="${CACHE_DIR:-$BASE_WORK/signal-analysis-cache/$RUN_ID}"

mkdir -p "$OUT_DIR" "$CACHE_DIR"/{hf,datasets,hub}

export HF_HOME="$CACHE_DIR/hf"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/hub"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

#   export HF_TOKEN=...
unset SSL_CERT_FILE SSL_CERT_DIR || true

echo "REPO_ROOT=$REPO_ROOT"
echo "MODEL=$MODEL"
echo "CASES_JSON=$CASES_JSON"
echo "N_CASES=$N_CASES  SEED=$SEED"
echo "OUT_DIR=$OUT_DIR"
echo

run "python \"$SCRIPT\" \
  --model \"$MODEL\" \
  --cases_json \"$CASES_JSON\" \
  --out_dir \"$OUT_DIR\" \
  --n_cases \"$N_CASES\" \
  --seed \"$SEED\""
