---
date: 2026-05-09
beads: [Sylveste-k8c, Sylveste-bvh, Sylveste-bov, Sylveste-2ss]
status: design
---

# Local-inference backend for flux-review agents

## Question

How many flux-review agents (`/flux-drive` typically dispatches 8–12) can we run in parallel using local inference on the M5 Max instead of paying for Claude Opus? What architecture, what model, what wall-clock?

## Two architectures (only one is good)

### Regime A — N processes, each loads the model

```
12 agents × 6 GB resident weights × 12 expert caches × 12 Metal contexts
≈ 72 GB duplicated weights + SSD thrash + GPU time-share
```

Apple Silicon GPU serializes Metal command submission across processes, so concurrent inference processes don't actually parallelize on the GPU — they timeshare. SSD bandwidth fights between expert caches. Practical ceiling: 3–5 concurrent before per-agent throughput collapses.

**Don't do this.**

### Regime B — One server, N concurrent requests

```
1 model loaded once (~6 GB resident + ~30 GB expert cache active set)
+ 12 in-flight POSTs → /v1/chat/completions
+ 1 GPU running batched matmuls as one wide kernel
```

The interfer server (`uv run python -m server`, port 8421) supports this today; never load-tested at 8+ concurrency.

## Throughput math

Qwen3.6-35B-A3B-DWQ (current C2 tier):

| Batch | KV cache @ 8K ctx | Per-request decode | Aggregate |
|---|---|---|---|
| 1 | ~2 GB | 5 tok/s | 5 |
| 4 | ~8 GB | 3 tok/s | 12 |
| 8 | ~16 GB | 2 tok/s | 16 |
| 16 | ~32 GB | 1.3 tok/s | 21 |
| 32 | ~64 GB | drops sharply | ~25 |

For a typical flux-review agent (10K input, 2K output): at batch=8, ~5 min wall per agent. Ten agents fanned out concurrently → ~5 min total wall (limited by the slowest).

vs. cloud Opus: ~30 sec wall but $1–2 per review session.

## Quality (open question)

Qwen3.6-35B-A3B-DWQ scores 40% on LCB v6 vs Opus 4.7 at ~88+%. Code-review-as-classification is an easier task than code-generation, so the gap may be smaller. Never measured in this repo. **Calibration run is the gate.**

## Better candidate when ready

DeepSeek-V4-Flash-4bit MLX (151 GB on disk, 13B active) is downloading right now under Sylveste-bvh. ~80% LCB v6 ceiling. Substantially better quality match for review work — the cost is slower per-request decode because of the SSD-streamed expert cache: probably 4 concurrent at ~8 min wall vs 8 concurrent at ~5 min on Qwen3.6.

## Scope (Sylveste-k8c)

1. **Bridge tool**: small wrapper that intercepts Agent() subagent dispatches and routes to `http://localhost:8421/v1/chat/completions` when `FLUX_LOCAL_BACKEND=interfer` is set. Pass-through to remote when not set.
2. **Server batch verification**: confirm interfer server handles 8+ concurrent requests cleanly.
3. **Agent prompt slimming**: flux-agent system prompts are sized for Opus context handling; trim to <2K tokens for a 35B-class model.
4. **Calibration suite**: 3–5 historical PRs with their Claude reviews, scored for findings-overlap, false-positive rate, severity calibration.
5. **A/B harness**: same diff routed to (a) Claude Opus via current path, (b) Qwen3.6-35B-A3B-DWQ local, (c) DeepSeek V4 Flash MLX local. Measure wall-clock, finding overlap, novel-finding rate.
6. **Decision gate**: if local-V4-Flash hits >75% finding-overlap with Opus on calibration set, promote local as default for routine reviews; reserve Opus for high-stakes work.

## Decision

Wait until Sylveste-bvh's V4 Flash MLX download completes (~3-4h from 2026-05-09 18:18 PDT) before running the actual experiment. Reasons:

- Running calibration on Qwen3.6-35B-A3B-DWQ would give a likely-false-negative ("local doesn't follow flux-agent prompts well enough") that wouldn't generalize to the model we'd actually want to use.
- V4 Flash 4-bit at 13B active is the first local model in this repo with credible code-quality numbers; it's the right candidate to test against.
- If V4 Flash quality is good but wall-clock is unworkable, fall back to Qwen as the speed-tier and reserve V4 for higher-stakes diffs.

## Out of scope (separate beads if needed)

- Replacing main Claude Code chat with local inference (different problem; chat needs steerability 35B-class can't reliably do)
- Fine-tuning a local model on Claude review traces (LoRA adapter from real review outputs — its own bead)
- Multi-tenant / shared-server deployment (this is single-user laptop tooling)
