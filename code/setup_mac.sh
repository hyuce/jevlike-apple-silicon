#!/usr/bin/env bash
# Idempotent setup of the Jev-like-on-Apple-Silicon environment, run by the user in macOS Terminal (full GPU access).
#
#   bash setup_mac.sh            # clone pinned repos, build ~/jevlike_cache/venv, write results/environment.md
#   bash setup_mac.sh --dry-run  # print what would happen, change nothing
#
# Everything heavy lives in $JEVLIKE_CACHE (default ~/jevlike_cache): repos/, venv/, hf/ (HF_HOME), models/.
# Google Drive receives only environment.md and the setup log. No model weights are downloaded here
# (run_smoke.sh does that through prepare_models.py).
set -euo pipefail

DRY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${JEVLIKE_RESULTS:-$PROJECT_DIR/results}"   # JEVLIKE_RESULTS: self-tests only
CACHE="${JEVLIKE_CACHE:-$HOME/jevlike_cache}"
VENV="$CACHE/venv"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$RESULTS_DIR/logs/setup_$STAMP.log"

# --- pinned repositories (same SHAs as pins.py; TMP/test_harness_cpu.py checks they agree) ---
REPO_NAMES=(kev laya nanojev jevmlx anyjev laya-mlx)
REPO_URLS=(
  "https://github.com/jaredpalmer/kev.git"
  "https://github.com/NandhaKishorM/laya.git"
  "https://github.com/TianyuCodings/NanoJev.git"
  "https://github.com/bnsd55/jevmlx.git"
  "https://github.com/nokia-applied-research/AnyJev.git"
  "https://github.com/mizorewww/laya-mlx.git"
)
REPO_SHAS=(
  "2074cc44256200bb37540103fa4572901faa10f3"
  "3a38cbe83f6ef556f3125fb0065fdc5a55d05792"
  "45b629283ddeb3b3784806f352bbf5e97f71dd40"
  "5f94ae262c1cf4f226b436433946be281ec1ac99"
  "35311c58955716cb8955ce1368a665988bc63735"
  "2159c6c9988e4cbfd6281db1888cef45de09cea9"
)

say() { echo "[setup] $*"; if [ "$DRY" = 0 ]; then echo "[setup] $*" >> "$LOG"; fi; }
run() {  # print, then execute unless --dry-run; output goes to the terminal and the log
  say "\$ $*"
  if [ "$DRY" = 0 ]; then "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; fi
}

# --- guards -------------------------------------------------------------------------------------------------
case "$CACHE" in
  *CloudStorage*|*GoogleDrive*|*"Google Drive"*)
    echo "JEVLIKE_CACHE=$CACHE is inside Google Drive; weights and clones must stay on the local disk." >&2; exit 3 ;;
esac
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "This setup targets Apple Silicon macOS (found $(uname -s) $(uname -m))." >&2; exit 3
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git not found. Install the Xcode command line tools:  xcode-select --install" >&2; exit 4
fi
if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
uv not found. Install it yourself (this script does not pipe installers into a shell), for example:
    brew install uv
or follow https://docs.astral.sh/uv/getting-started/installation/ , open a new Terminal, then rerun:
    bash setup_mac.sh
EOF
  exit 5
fi

if [ "$DRY" = 0 ]; then
  mkdir -p "$CACHE/repos" "$CACHE/hf" "$CACHE/models" "$RESULTS_DIR/logs"
  CACHE="$(cd "$CACHE" && pwd)"; VENV="$CACHE/venv"      # absolute from here on
fi
export HF_HOME="$CACHE/hf"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE/uv-cache}"
export GIT_LFS_SKIP_SMUDGE=1
say "project:  $PROJECT_DIR"
say "cache:    $CACHE   (HF_HOME=$HF_HOME, UV_CACHE_DIR=$UV_CACHE_DIR)"
say "log:      $LOG"
say "uv:       $(uv --version)   git: $(git --version)"
if [ "$DRY" = 0 ]; then
  say "disk free at cache: $(df -h "$CACHE" | awk 'NR==2{print $4}')  (setup ~5 GB; run_smoke.sh then ~35 GB of weights)"
fi

# --- 1. repositories at pinned commits ----------------------------------------------------------------------
clone_at() {  # name url sha
  local name="$1" url="$2" sha="$3" dir="$CACHE/repos/$1"
  if [ -d "$dir/.git" ]; then
    local have; have="$(git -C "$dir" rev-parse HEAD 2>/dev/null || echo none)"
    if [ "$have" = "$sha" ]; then say "repo $name already at $sha"; return 0; fi
    if [ -n "$(git -C "$dir" status --porcelain 2>/dev/null)" ]; then
      echo "repo $name at $dir has local changes; not touching it. Resolve by hand, then rerun." >&2; exit 6
    fi
    run git -C "$dir" fetch --depth 1 origin "$sha"
    run git -C "$dir" checkout --detach FETCH_HEAD
  else
    run git init -q "$dir"
    run git -C "$dir" remote add origin "$url"
    run git -C "$dir" fetch --depth 1 origin "$sha"
    run git -C "$dir" checkout -q --detach FETCH_HEAD
  fi
  if [ "$DRY" = 0 ]; then
    local now; now="$(git -C "$dir" rev-parse HEAD)"
    if [ "$now" != "$sha" ]; then echo "repo $name ended at $now, expected $sha" >&2; exit 7; fi
    say "repo $name at $now"
  fi
}
for i in "${!REPO_NAMES[@]}"; do clone_at "${REPO_NAMES[$i]}" "${REPO_URLS[$i]}" "${REPO_SHAS[$i]}"; done

# --- 2. venv: Kev's locked environment (+ serve extra), then the other candidates under Kev's pins --------------
# uv sync --frozen reproduces kev's uv.lock exactly (torch 2.8.0, transformers 5.17.0, mlx 0.32.2, mlx-lm 0.31.3 at
# kev 2074cc44); --inexact keeps the packages added in the next step on reruns.
export UV_PROJECT_ENVIRONMENT="$VENV"
run uv sync --project "$CACHE/repos/kev" --extra serve --frozen --inexact --python 3.12
run uv export --project "$CACHE/repos/kev" --frozen --extra serve --no-hashes --no-emit-project --no-dev -o "$CACHE/kev_constraints.txt"
# laya, laya-mlx, AnyJev (transformers backend) and jevmlx resolved WITH kev's versions as constraints (checked with
# `uv pip compile` on 2026-11-13: adds vegaml, cryptography, ...; changes none of kev's pins).
run uv pip install --python "$VENV/bin/python" -c "$CACHE/kev_constraints.txt" \
  -e "$CACHE/repos/laya" -e "$CACHE/repos/laya-mlx" -e "$CACHE/repos/anyjev[hf]" -e "$CACHE/repos/jevmlx" psutil pandas scipy
say "\$ uv pip freeze --python $VENV/bin/python > $RESULTS_DIR/logs/venv_freeze_$STAMP.txt"
if [ "$DRY" = 0 ]; then uv pip freeze --python "$VENV/bin/python" > "$RESULTS_DIR/logs/venv_freeze_$STAMP.txt" || true; fi

# --- 3. environment record -------------------------------------------------------------------------------------
run "$VENV/bin/python" "$SCRIPT_DIR/env_report.py" --out "$RESULTS_DIR/environment.md"
if [ "$DRY" = 0 ]; then
  "$VENV/bin/python" - <<'EOF' 2>&1 | tee -a "$LOG"
import mlx.core as mx, torch
print("[setup] check: MLX Metal available =", mx.metal.is_available(), "| torch MPS available =", torch.backends.mps.is_available())
EOF
fi
say "done. Next:  bash \"$SCRIPT_DIR/run_smoke.sh\""
