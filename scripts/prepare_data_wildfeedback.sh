#!/usr/bin/env bash
set -euo pipefail

# Prepare offline training data from microsoft/WildFeedback.
#
# Usage:
#   ./scripts/prepare_data_wildfeedback.sh [--dry-run]
#
# Common overrides:
#   OUT_DIR=/path/to/wildfeedback_data ./scripts/prepare_data_wildfeedback.sh
#
# Output:
#   $OUT_DIR/wildfeedback_interactions.jsonl

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
PREP_SCRIPT="${PREP_SCRIPT:-$REPO_ROOT/auxiliary/preprocess_wildfeedback.py}"

# =============================================================================
# Configuration
# =============================================================================
OUT_DIR="${OUT_DIR:-$REPO_ROOT/data/wildfeedback}"

# Optional HF cache dir
CACHE_DIR="${CACHE_DIR:-}"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ -n "$CACHE_DIR" ]]; then
  mkdir -p "$CACHE_DIR"/{hf,datasets}
  export HF_HOME="$CACHE_DIR/hf"
  export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
fi

#   export HF_TOKEN=...
unset SSL_CERT_FILE SSL_CERT_DIR || true

echo "REPO_ROOT=$REPO_ROOT"
echo "OUT_DIR=$OUT_DIR"
echo

run "python \"$PREP_SCRIPT\" --out_dir \"$OUT_DIR\""
