---
artifact_type: brainstorm
bead: sylveste-2je
stage: brainstorm
---

# Brainstorm: Wire Flash-MoE into End-to-End Live Serving

**Bead:** sylveste-2je
**Date:** 2026-04-05

## Current State

The FlashMoeWorker and server integration is ~85% implemented:

- **FlashMoeWorker** (`server/flashmoe_worker.py`): HTTP-based subprocess proxy with watchdog, crash recovery, generate lock — fully functional
- **Server routing** (`server/main.py:509-522`): Model-name matching routes to `_generate_flashmoe_tokens()` — implemented
- **CLI flags** (`server/__main__.py:112-185`): All flash-moe flags plumbed (q3, cache-io-split, gguf overlays) — implemented
- **Lifecycle** (`server/main.py:857-859`): FlashMoeWorker starts on app startup, shuts down on app teardown — implemented
- **SSE streaming** (`server/main.py:236-295`): `_generate_flashmoe_tokens()` wraps synchronous generator in asyncio executor — implemented

## What's Broken

### 1. Test suite (16 failures)

The `FlashMoeWorker.__init__` signature was refactored to remove `enable_watchdog` parameter, but all test constructors still pass `enable_watchdog=False`. Three categories:

**a) Constructor mismatch (all 16 failures):** Every test creates workers with `enable_watchdog=False` — needs removal.

**b) Health response format mismatch:** Tests expect `{"status": "ready", "backend": "flash-moe", "loaded_models": [...]}` but actual `health()` returns `{"status": "ok", "port": N, ...raw_data}`. Tests need to match actual behavior.

**c) Missing properties:** Tests reference `is_restarting`, `restart_count`, `last_crash`, `crash_history` — none exist on the class. Either add them or remove tests.

**d) Generate signature:** Tests call `w.generate(messages=[...])` without `model_name` — but `generate()` requires `model_name` as first positional arg.

### 2. No end-to-end smoke test

Nothing validates: `curl → server → FlashMoeWorker → flash-moe binary → streamed tokens`. The unit tests use a fake HTTP server, which is good for isolation but doesn't test the real binary path.

### 3. Watchdog always starts

With `enable_watchdog` removed, `start()` always spawns the watchdog thread. Tests that create workers without real binaries will have watchdog threads trying to restart nonexistent processes. Need a way to suppress this in tests (mock, or check binary exists before restart).

## Approach

**Fix the test suite to match the current FlashMoeWorker API.** This is a test-update task, not a code rewrite. The worker implementation is correct; the tests are stale.

### Changes needed:

1. Remove `enable_watchdog=False` from all test constructors
2. Update health assertions to match actual response format
3. Add missing properties to FlashMoeWorker: `is_restarting`, `restart_count`, `last_crash`, `crash_history` (4 simple properties over existing state)
4. Fix `generate()` calls to pass `model_name` positional arg
5. Add one integration test that starts the server with `--flashmoe-only --dry-run` equivalent (mock the binary, verify HTTP round-trip)

### What this does NOT change:

- No server code changes — the routing, streaming, lifecycle are correct
- No FlashMoeWorker subprocess logic changes
- No CLI flag changes

## Success Criteria

- All 27 tests pass (16 currently failing + 11 passing)
- `uv run python -m server --flashmoe-only --flashmoe-binary <path> --flashmoe-model <path>` starts and serves tokens
- `curl localhost:8421/v1/chat/completions -d '{"model":"flash-moe","messages":[...],"stream":true}'` returns SSE tokens

## Non-Goals

- Benchmark throughput (separate bead: sylveste-e25)
- Track B5 integration (separate bead: sylveste-5ji)
- New features or optimizations
