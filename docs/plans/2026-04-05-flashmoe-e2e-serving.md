---
artifact_type: plan
bead: sylveste-2je
stage: planned
---

# Plan: Wire Flash-MoE into End-to-End Live Serving

**Bead:** sylveste-2je
**Date:** 2026-04-05

## Task 1: Add missing diagnostic properties to FlashMoeWorker

**File:** `server/flashmoe_worker.py`

Add 4 properties over existing state:
- `is_restarting` → `bool`: True during watchdog restart backoff (add `_is_restarting` flag, set in `_watchdog_loop`)
- `restart_count` → `int`: Return `self._restart_count`
- `last_crash` → `float | None`: Timestamp of last crash (add `_last_crash_time` field)
- `crash_history` → `list[dict]`: List of `{time, exit_code, consecutive}` (add `_crash_history` list)

**Lines of code:** ~30 additions
**Risk:** Low — additive only, no existing behavior changes

## Task 2: Fix test constructors — remove `enable_watchdog=False`

**File:** `tests/test_flashmoe_worker.py`

Remove `enable_watchdog=False` from every `FlashMoeWorker(...)` call in tests. Since tests don't call `start()` on most workers (they mock `_process` directly), the watchdog won't start. For tests that do call `start()` with fake binaries, they'll fail before `_start_watchdog()` is reached.

**Affected:** Lines 128-133, 187-190, 196-199, 212-216, 224-228, 232-237, 245-249, 256-261, 300-305, 312-317, 322-327

## Task 3: Fix health response assertions

**File:** `tests/test_flashmoe_worker.py`

Current `health()` returns `{"status": "ok", "port": N, ...raw_upstream}`. Tests expect `{"status": "ready", "backend": "flash-moe", "loaded_models": [...]}`.

Options:
- **A)** Update tests to match current behavior → simplest
- **B)** Update `health()` to normalize response → adds value for monitoring

**Choose B** — normalize in `health()` to return a consistent shape regardless of upstream format. Add `backend`, `loaded_models` fields. This benefits the `/health` endpoint too.

## Task 4: Fix generate() calls — add `model_name` parameter

**File:** `tests/test_flashmoe_worker.py`

Add `model_name="flash-moe"` to all `w.generate(...)` calls that omit it.

Alternatively, make `model_name` optional with default `"flash-moe"` in the worker — this is more ergonomic since the worker already knows what model it's serving.

**Choose:** Make `model_name` optional with default `"flash-moe"`.

## Task 5: Verify all tests pass

Run `uv run pytest tests/test_flashmoe_worker.py -v` and fix any remaining issues.

## Task 6: Manual smoke test documentation

Add a comment block or section in AGENTS.md documenting the manual e2e test command:

```bash
uv run python -m server \
  --flashmoe-only \
  --flashmoe-binary ~/projects/flash-moe/metal_infer/infer \
  --flashmoe-model ~/Models/flash_mlx_4bit \
  --flashmoe-q3-experts \
  --flashmoe-cache-io-split 4
```

## Execution Order

Tasks 1-4 are independent (parallel-safe). Task 5 depends on 1-4. Task 6 depends on 5.
