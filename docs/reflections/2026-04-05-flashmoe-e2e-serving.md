---
artifact_type: reflection
bead: sylveste-2je
stage: reflect
---

# Reflection: Wire Flash-MoE into End-to-End Live Serving

**Bead:** sylveste-2je
**Date:** 2026-04-05

## What happened

The brainstorm revealed the integration was ~85% done — far more complete than the vision doc suggested. The actual gap was a stale test suite (16/27 failures) from a refactoring that removed the `enable_watchdog` parameter, changed health response format, and shifted the `generate()` signature.

## What we learned

1. **Test-code drift is the real blocker.** The server code was correct and working. The 16 test failures made it _look_ broken, which led to this being scoped as a "wire up integration" task when it was really "align tests with existing implementation." Reading the actual code first (before planning) saved significant time.

2. **Normalize at the boundary.** Adding a normalization layer in `health()` that always returns `{status, backend, port, loaded_models}` decouples the server from upstream binary response format changes. This is cheap insurance — the flash-moe binary is actively developed upstream and its response format is not stable.

3. **Self-instrumentation in generate() is worth the ~5 lines.** Tracking `tokens_generated`, computing local `generation_tps` when upstream doesn't report usage stats, and tagging `backend` means monitoring works regardless of binary version. The alternative — requiring a specific binary version — would be a constant maintenance burden.

## What we'd do differently

The initial bead title ("Wire Flash-MoE into end-to-end live serving") implied significant integration work. A quick code read at bead-creation time would have right-sized it to "Fix FlashMoeWorker test suite + add diagnostics." This matters for prioritization — a test-fix P2 might have been deferred, while a "wire up integration" P1 got sprinted immediately.
