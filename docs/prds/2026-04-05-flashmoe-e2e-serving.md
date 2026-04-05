---
artifact_type: prd
bead: sylveste-2je
stage: strategy
---

# PRD: Wire Flash-MoE into End-to-End Live Serving

**Version:** 1.0
**Date:** 2026-04-05
**Bead:** sylveste-2je

## Problem

FlashMoeWorker implementation is complete but its test suite has 16/27 failures from a stale API contract (`enable_watchdog` removed, health response format changed, missing properties). The worker also lacks a few diagnostic properties that tests (and future monitoring) need. Until tests pass, we can't confidently claim end-to-end serving works.

## Features

### F1: Fix FlashMoeWorker Test Suite

Update all test constructors and assertions to match the current FlashMoeWorker API.

**Acceptance:** 27/27 tests pass.

### F2: Add Diagnostic Properties

Add `is_restarting`, `restart_count`, `last_crash`, `crash_history` properties to FlashMoeWorker over existing state fields.

**Acceptance:** Properties return correct values; covered by existing (updated) tests.

### F3: Generate Signature Alignment

Fix test `generate()` calls to pass required `model_name` parameter.

**Acceptance:** All generate-related tests pass without `TypeError`.

## Non-Goals

- No server code changes
- No new CLI flags
- No throughput benchmarking
- No Track B5 integration

## Success Metrics

| Metric | Target |
|--------|--------|
| Test pass rate | 27/27 |
| Server startup with --flashmoe-only | Starts without error |
| curl chat/completions | Returns SSE tokens (manual verification with real binary) |
