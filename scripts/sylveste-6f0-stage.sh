#!/usr/bin/env bash
# Sylveste-6f0 staging driver: validate the flash-moe wedge fix incrementally.
#
# Runs LCB v6 against flash-moe at progressively larger --limit values
# (1 → 5 → 50 → 150 → 175). After each stage, evaluates machine-checkable
# tripwires; if any fail, halts and writes forensic state for inspection.
# If all pass, auto-promotes to the next stage.
#
# The bug is state-accumulating (wedge first observed at problem ~141 in
# the original LCB v6 matrix), so each stage answers a different question:
#
#   S0 (1):    APIs load; cancel + drainer at least don't break happy path
#   S1 (5):    Multiple completions; stderr drainer accumulates bytes
#   S2 (50):   Sustained operation; fd count stable; no early wedge
#   S3 (150):  Crosses historical wedge zone (~141) — the real test
#   S4 (175):  Apples-to-apples completion of LCB v6 matrix
#
# Tripwires (all post-stage, machine-checkable):
#   T1: zero "urlopen.*Errno 60" in run.log         — H2 cascade gone
#   T2: timeouts ≤ max(1, ceil(0.03 × stage_N))     — aggressive ceiling
#   T3: stderr log file exists and has bytes        — H1 drainer working
#   T4: flashmoe pid still alive (kill -0)          — basic liveness
#
# On failure: forensics dumped to lcb_v6_matrix/sylveste-6f0-stage-S{n}-fail/
# with run.log tail, lsof, ps, and stderr log copy.
#
# Usage:
#   bash scripts/sylveste-6f0-stage.sh                         # default ladder
#   STAGE_LADDER="1 5 50" bash scripts/sylveste-6f0-stage.sh   # custom ladder
#   OUT=/tmp/6f0 bash scripts/sylveste-6f0-stage.sh            # alt output

set -euo pipefail

OUT="${OUT:-benchmarks/lcb_v6_matrix}"
MODEL="${MODEL:-flash-moe:397b}"
TIMEOUT="${TIMEOUT:-180}"
STAGE_LADDER="${STAGE_LADDER:-1 5 50 150 175}"
STDERR_DIR="${HOME}/.cache/interfer"
DRY_RUN_FLAG="${DRY_RUN:-}"

mkdir -p "$OUT"
touch "$OUT/run.log"

log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*"; }

# Compute the hard-fail timeout ceiling for a given stage size.
#   max(1, ceil(0.03 × N))
fail_threshold() {
  local n="$1"
  python3 -c "import math; print(max(1, math.ceil(0.03 * $n)))"
}

# Count NEW lines added to run.log during this stage. We snapshot wc -l
# before each stage and diff against post-stage wc -l.
run_log="$OUT/run.log"

# Find the most recent flash-moe stderr file (pid-named) and its size.
# Returns "PID SIZE" or "0 0" if none.
flashmoe_stderr_status() {
  if [[ ! -d "$STDERR_DIR" ]]; then
    echo "0 0"
    return
  fi
  local newest
  newest=$(ls -t "$STDERR_DIR"/flashmoe-*.stderr 2>/dev/null | head -1)
  if [[ -z "$newest" ]]; then
    echo "0 0"
    return
  fi
  local pid size
  pid=$(basename "$newest" | sed -E 's/flashmoe-([0-9]+)\.stderr/\1/')
  size=$(stat -f %z "$newest" 2>/dev/null || echo 0)
  echo "$pid $size"
}

# Capture forensics on stage failure.
dump_forensics() {
  local stage="$1"
  local reason="$2"
  local dir="$OUT/sylveste-6f0-stage-S${stage}-fail"
  mkdir -p "$dir"
  log "DUMP forensics → $dir"
  echo "$reason" > "$dir/reason.txt"
  tail -n 200 "$run_log" > "$dir/run.log.tail" 2>/dev/null || true

  # Process state for the flash-moe binary if we can find it.
  read -r pid _size < <(flashmoe_stderr_status)
  if [[ "$pid" != "0" ]] && kill -0 "$pid" 2>/dev/null; then
    ps -p "$pid" -o pid,rss,vsz,etime,command > "$dir/ps.txt" 2>/dev/null || true
    lsof -p "$pid" 2>/dev/null > "$dir/lsof.txt" || true
    cp "$STDERR_DIR/flashmoe-$pid.stderr" "$dir/flashmoe.stderr" 2>/dev/null || true
  else
    echo "no live flashmoe pid (last known: $pid)" > "$dir/ps.txt"
    if [[ "$pid" != "0" ]]; then
      cp "$STDERR_DIR/flashmoe-$pid.stderr" "$dir/flashmoe.stderr" 2>/dev/null || true
    fi
  fi
}

# Evaluate tripwires after a stage. Returns 0 if all green, 1 on first failure.
check_tripwires() {
  local stage_idx="$1"
  local stage_n="$2"
  local pre_log_lines="$3"
  local pre_stderr_size="$4"
  local pre_pid="${5:-0}"

  local post_log_lines
  post_log_lines=$(wc -l < "$run_log" 2>/dev/null | tr -d ' ' || echo 0)
  post_log_lines=${post_log_lines:-0}
  local new_lines=$((post_log_lines - pre_log_lines))

  # Slice out just this stage's run.log lines.
  local stage_log
  stage_log=$(mktemp)
  tail -n "$new_lines" "$run_log" > "$stage_log" 2>/dev/null || true

  # T1: zero "urlopen [Errno 60]" — H2 cascade.
  # Use grep + wc -l to avoid grep -c quirks (multiple zero outputs on
  # empty input, exit code on no-match).
  local cascade_count
  cascade_count=$(grep 'urlopen.*Errno 60' "$stage_log" 2>/dev/null | wc -l | tr -d ' ')
  cascade_count=${cascade_count:-0}
  if (( cascade_count > 0 )); then
    log "TRIPWIRE T1 FAIL: $cascade_count 'urlopen Errno 60' errors (H2 lock-leak cascade)"
    rm -f "$stage_log"
    dump_forensics "$stage_idx" "T1: urlopen Errno 60 cascade ($cascade_count occurrences)"
    return 1
  fi

  # T2: timeouts ≤ ceil(0.03 × N)
  local timeout_count threshold
  timeout_count=$(grep 'GENERATION ERROR: timed out' "$stage_log" 2>/dev/null | wc -l | tr -d ' ')
  timeout_count=${timeout_count:-0}
  threshold=$(fail_threshold "$stage_n")
  if (( timeout_count > threshold )); then
    log "TRIPWIRE T2 FAIL: $timeout_count timeouts > $threshold threshold"
    rm -f "$stage_log"
    dump_forensics "$stage_idx" "T2: $timeout_count timeouts exceeded threshold $threshold (3% of $stage_n)"
    return 1
  fi
  if (( timeout_count > 0 )); then
    local warn_threshold
    warn_threshold=$(python3 -c "import math; print(max(1, math.ceil(0.01 * $stage_n)))")
    if (( timeout_count > warn_threshold )); then
      log "TRIPWIRE T2 WARN: $timeout_count timeouts > $warn_threshold soft threshold"
    fi
  fi

  rm -f "$stage_log"

  # T3 + T4 only apply to flash-moe models (cloud/mlx don't spawn the binary).
  if [[ "$MODEL" != *flash-moe* ]] && [[ "$MODEL" != *flashmoe* ]]; then
    log "T3/T4 skipped: MODEL=$MODEL is not a flash-moe backend"
    return 0
  fi

  # T3: stderr log file exists and grew. Note pre_stderr_size was captured
  # against whatever pid was newest at stage-start; if the worker restarted
  # mid-stage (watchdog SIGKILL → restart), the newest file now has a
  # different pid. In that case "did it grow" is the wrong question — we
  # care that *some* stderr file exists with bytes from this stage.
  local pid post_stderr_size
  read -r pid post_stderr_size < <(flashmoe_stderr_status)
  if [[ "$pid" == "0" ]]; then
    log "TRIPWIRE T3 FAIL: no stderr log file under $STDERR_DIR (H1 drainer didn't run)"
    dump_forensics "$stage_idx" "T3: stderr drainer never created a log file"
    return 1
  fi
  if [[ "$pid" != "$pre_pid" ]] && [[ "$pre_pid" != "0" ]]; then
    log "T3 note: worker restarted mid-stage ($pre_pid → $pid)"
    if (( post_stderr_size > 0 )); then
      log "T3 ok: new worker stderr=$post_stderr_size bytes (pid $pid)"
    else
      log "TRIPWIRE T3 WARN: new worker has empty stderr file"
    fi
  elif (( post_stderr_size <= pre_stderr_size )); then
    log "TRIPWIRE T3 WARN: stderr file did not grow this stage ($pre_stderr_size → $post_stderr_size bytes)"
    log "  (acceptable on S0 if binary writes nothing yet; concerning on later stages)"
  else
    local delta=$((post_stderr_size - pre_stderr_size))
    log "T3 ok: stderr grew by $delta bytes (pid $pid)"
  fi

  # T4: flashmoe pid still alive (only if we have one — S0 might exit cleanly)
  if [[ "$pid" != "0" ]] && ! kill -0 "$pid" 2>/dev/null; then
    log "TRIPWIRE T4 FAIL: flashmoe pid $pid is no longer alive"
    dump_forensics "$stage_idx" "T4: flashmoe pid $pid died (watchdog should have restarted, but matrix harness exits between models)"
    return 1
  fi
  if [[ "$pid" != "0" ]]; then
    log "T4 ok: flashmoe pid $pid alive"
  fi

  return 0
}

run_stage() {
  local stage_idx="$1"
  local stage_n="$2"

  log "==============================================="
  log "  Stage S$stage_idx — limit=$stage_n"
  log "==============================================="

  local pre_log_lines pre_stderr_size pre_pid
  pre_log_lines=$(wc -l < "$run_log" 2>/dev/null || echo 0)
  read -r pre_pid pre_stderr_size < <(flashmoe_stderr_status)
  log "pre: run.log lines=$pre_log_lines  flashmoe pid=$pre_pid  stderr=$pre_stderr_size bytes"

  # Run the harness. Output is teed into run.log so the matrix's existing
  # log layout is preserved.
  local dry_args=()
  [[ -n "$DRY_RUN_FLAG" ]] && dry_args=(--dry-run)
  if ! uv run python -m benchmarks.code_correctness \
        --model="$MODEL" \
        --suite=livecodebench-v6 \
        --output="$OUT" \
        --timeout="$TIMEOUT" \
        --limit="$stage_n" \
        "${dry_args[@]}" 2>&1 | tee -a "$run_log"; then
    log "STAGE S$stage_idx HARNESS EXITED NON-ZERO"
    dump_forensics "$stage_idx" "harness exited non-zero (uv run returned failure)"
    return 1
  fi

  if ! check_tripwires "$stage_idx" "$stage_n" "$pre_log_lines" "$pre_stderr_size" "$pre_pid"; then
    log "STAGE S$stage_idx FAILED — halting ladder"
    return 1
  fi

  log "STAGE S$stage_idx PASSED"
  return 0
}

main() {
  log "Sylveste-6f0 staging — model=$MODEL  out=$OUT  ladder=[$STAGE_LADDER]"
  log "Tripwires: T1 zero Errno-60, T2 timeouts ≤3% (warn ≥1%), T3 stderr grows, T4 pid alive"

  local idx=0
  for n in $STAGE_LADDER; do
    if ! run_stage "$idx" "$n"; then
      log "Ladder halted at S$idx (limit=$n). Inspect $OUT/sylveste-6f0-stage-S${idx}-fail/"
      exit 1
    fi
    idx=$((idx + 1))
  done

  log "==============================================="
  log "  All stages passed — $MODEL is wedge-free."
  log "==============================================="
}

main "$@"
