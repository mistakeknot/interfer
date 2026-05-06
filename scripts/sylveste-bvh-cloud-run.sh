#!/bin/bash
# Sylveste-bvh: run LCB v6 against a single cloud model alias.
#
# Usage:  bash scripts/sylveste-bvh-cloud-run.sh <model-alias>
# Example: bash scripts/sylveste-bvh-cloud-run.sh cloud:deepseek-v4-flash
#
# Reads OPENROUTER_API_KEY from ~/.cache/interfer/openrouter.key (mode 600).
# Streams progress to benchmarks/lcb_v6_matrix/sylveste-bvh-<alias>-<date>.log
# (line-buffered via stdbuf so cache JSONL grows visibly).

set -euo pipefail

MODEL_ALIAS="${1:-}"
if [[ -z "$MODEL_ALIAS" ]]; then
  echo "usage: $0 <model-alias>" >&2
  exit 2
fi

KEYFILE="$HOME/.cache/interfer/openrouter.key"
if [[ ! -f "$KEYFILE" ]]; then
  echo "missing $KEYFILE — drop OPENROUTER_API_KEY there (chmod 600)" >&2
  exit 2
fi

export OPENROUTER_API_KEY
OPENROUTER_API_KEY="$(cat "$KEYFILE")"

# Match the layout used by the prior wedge-fix run so artifacts live alongside.
cd /Users/sma/projects/Sylveste/interverse/interfer

OUT="benchmarks/lcb_v6_matrix"
SAFE_ALIAS="${MODEL_ALIAS//[:\/]/-}"
DATE="$(date -u +%Y-%m-%d)"
LOG="${OUT}/sylveste-bvh-${SAFE_ALIAS}-${DATE}.log"

mkdir -p "$OUT"

{
  echo "=== Sylveste-bvh cloud validation run ==="
  echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Model:    $MODEL_ALIAS"
  echo "Output:   $OUT"
  echo "Log:      $LOG"
  echo "uv:       $(command -v uv) ($(uv --version))"
  echo "==========================================="
  echo

  # Python's print(..., flush=True) is enough — don't pull in Homebrew stdbuf
  # (Homebrew coreutils' libstdbuf.so is x86_64 on some setups and won't load
  # into arm64 Python, killing the run before it starts).
  PYTHONUNBUFFERED=1 uv run python -m benchmarks.code_correctness \
    --model "$MODEL_ALIAS" \
    --suite livecodebench-v6 \
    --output "$OUT" \
    --timeout 600

  echo
  echo "==========================================="
  echo "Run complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$LOG" 2>&1
