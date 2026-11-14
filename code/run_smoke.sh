#!/usr/bin/env bash
# Prepare the pinned models (download, merge, convert) and run the smoke test, all in one command.
#
#   bash run_smoke.sh                      # prepare everything, then smoke-test every combination
#   bash run_smoke.sh --dry-run            # plan only: disk estimate, prerequisites, combinations; loads nothing
#   bash run_smoke.sh --smoke-only [...]   # skip preparation (models already prepared)
#   bash run_smoke.sh --only kev-0.8b/bf16,laya/fp32   # extra arguments go to smoke_test.py
#
# Needs setup_mac.sh first. Weights go to $JEVLIKE_CACHE (default ~/jevlike_cache); results to ../results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="${JEVLIKE_CACHE:-$HOME/jevlike_cache}"
PY="${JEVLIKE_PYTHON:-$CACHE/venv/bin/python}"

DRY=0; SMOKE_ONLY=0; PASS=(); PREP_PASS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --smoke-only) SMOKE_ONLY=1 ;;
    --tf-base=*) PASS+=("--tf-base" "${arg#*=}"); PREP_PASS+=("--tf-base" "${arg#*=}") ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) PASS+=("$arg") ;;
  esac
done

case "$CACHE" in
  *CloudStorage*|*GoogleDrive*|*"Google Drive"*) echo "JEVLIKE_CACHE must not be inside Google Drive" >&2; exit 3 ;;
esac
if [ ! -x "$PY" ]; then
  echo "python not found at $PY; run setup_mac.sh first (or set JEVLIKE_PYTHON)." >&2; exit 4
fi
export JEVLIKE_CACHE="$CACHE"
export HF_HOME="$CACHE/hf"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
# PYTORCH_ENABLE_MPS_FALLBACK is deliberately NOT set: an op MPS lacks must fail visibly, not run silently on the CPU.
unset PYTORCH_ENABLE_MPS_FALLBACK || true

cd "$SCRIPT_DIR"
if [ "$DRY" = 1 ]; then
  "$PY" prepare_models.py --dry-run ${PREP_PASS[@]+"${PREP_PASS[@]}"}
  "$PY" smoke_test.py --dry-run ${PASS[@]+"${PASS[@]}"}
  exit 0
fi
if [ "$SMOKE_ONLY" = 0 ]; then
  # a failed task does not stop the run: the combinations that need it are recorded as runs=no with the reason
  "$PY" prepare_models.py ${PREP_PASS[@]+"${PREP_PASS[@]}"} || echo "[run_smoke] prepare_models.py reported failures; continuing to the smoke test"
fi
"$PY" smoke_test.py ${PASS[@]+"${PASS[@]}"}
