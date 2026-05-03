#!/usr/bin/env bash
# Sylveste-6f0 direct validation: invoke code_correctness against flash-moe
# at full --limit=175. The harness skips cached rows (141 from the original
# 04-26 matrix), so this effectively re-runs only the wedge zone (problems
# 142-175 = 34 uncached) with the H1+H2+H3 fixes in place.
#
# Designed to be invokable by launchd (no shell-init dependencies).

set -euo pipefail

# launchd starts with a sparse PATH; restore the user-level binaries.
export PATH="/Users/sma/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
export HOME="/Users/sma"

cd /Users/sma/projects/Sylveste/interverse/interfer

OUT="benchmarks/lcb_v6_matrix"
LOG="${OUT}/sylveste-6f0-direct-2026-05-03.log"

mkdir -p "$OUT"

{
  echo "=== Sylveste-6f0 direct validation run ==="
  echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "PWD: $(pwd)"
  echo "uv: $(which uv) ($(uv --version))"
  echo "Cache pre-state:"
  python3 -c "
import json
flash = sum(1 for line in open('${OUT}/code_correctness.jsonl')
            if json.loads(line).get('model','').startswith('flash'))
print(f'  flash-moe rows already cached: {flash}/175')
print(f'  expected new generations: {175 - flash}')
"
  echo "==========================================="
  echo

  uv run python -m benchmarks.code_correctness \
    --model=flash-moe:397b \
    --suite=livecodebench-v6 \
    --output="$OUT" \
    --timeout=180 \
    --limit=175

  echo
  echo "==========================================="
  echo "Run complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Cache post-state:"
  python3 -c "
import json
flash = sum(1 for line in open('${OUT}/code_correctness.jsonl')
            if json.loads(line).get('model','').startswith('flash'))
print(f'  flash-moe rows now: {flash}/175')
"
  echo "Stderr drainage check:"
  ls -la ~/.cache/interfer/flashmoe-*.stderr 2>&1 | tail -5 || echo "  (no stderr files)"
} >> "$LOG" 2>&1
