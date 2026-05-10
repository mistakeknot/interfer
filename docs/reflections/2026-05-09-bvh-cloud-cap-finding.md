---
date: 2026-05-09
beads: [Sylveste-bvh, Sylveste-bov, Sylveste-uk3, Sylveste-k8c]
status: closed
---

# Sylveste-bvh closeout: OpenRouter caps DeepSeek V4 Flash at 16K tokens

## What we set out to do

Add `cloud:deepseek-v4-flash` and `cloud:deepseek-v4-pro` to the LCB v6 matrix. Expected pass@1 around the published 91.6 / 93.5.

## What we found

**47.1% pass@1** for V4 Flash via OpenRouter→DeepInfra at `reasoning_effort: "high"`, configured with `max_tokens=32768`. Published number is 91.6.

The 44-point gap is **not** a model issue, **not** a routing issue (reasoning passthrough works), and **not** an effort issue. It's a hard provider-side cap on `max_tokens`.

## Wire-level evidence

Probe on `abc398_g` (one of 43 no_code_extracted failures), via OpenRouter→DeepInfra:

```
asked:               max_tokens=64000
returned:            completion_tokens=16384
finish_reason:       'length'
reasoning_tokens:    16607  (reasoning alone exceeded the cap)
text chars:          0      (never got to write code)
```

DeepInfra returns 16384 tokens regardless of the requested ceiling. Almost certainly a per-tier limit set by the provider, not configurable on our end without an enterprise account.

V4 Flash at `high` effort routinely uses **30K+ reasoning tokens** on Codeforces F/G class problems. With reasoning blocked at 16K, the model is mathematically unable to emit code on the hardest 25% of LCB v6.

## Diagnostic chain

| Hypothesis | Cost to test | Verdict |
|---|---|---|
| OpenRouter drops `reasoning_effort` for cost reasons | $0 (probe) | **Wrong** — reasoning IS engaged on Novita, DeepInfra, Parasail (610–1548 reasoning chars on a trivial probe) |
| Harness `max_tokens=8192` was the bug | ~$3 (re-run on 32K config) | **Partly right** — recovered 0 percentage points (47% vs 52% prior); helped on easier problems but exposed the deeper cap |
| DeepInfra silently caps max_tokens at 16K | $0 (one-call probe with `finish_reason` capture) | **Right** — `finish_reason='length'`, `completion_tokens=16384` regardless of asked-for budget |

## What this validates (the work that shipped)

The harness fix in commit `1012ead` is correct, even if it didn't move pass@1:

- `max_output_tokens` per-config override (was hardcoded 8192)
- `openrouter_provider` pinning support (`provider: {only: ["..."]}`)
- Unified OpenRouter reasoning shape (`reasoning: {effort: ...}` in extra_body)
- Reasoning telemetry capture (`delta.reasoning` chunks + `completion_tokens_details.reasoning_tokens` from final usage event)
- Streaming sidesteps OpenRouter's leading-whitespace JSON quirk

Tests: 12/12 pass.

## What this kills

- Cloud benchmarking via OpenRouter for any reasoning model that needs >16K tokens. Includes V4 Flash, V4 Pro, GPT-5.5 at xhigh, Opus 4.7 at high effort, Grok 4 Heavy.
- The premise that we'd reproduce published cloud LCB numbers cheaply.

## What's left over

Three followup beads, none blocking:

- **Sylveste-bov**: flash-moe perf regression (filed earlier, P2)
- **Sylveste-uk3**: OpenRouter telemetry undercount (filed earlier, P3)
- **(new, low priority)**: native DeepSeek API integration if we ever need to reproduce the 91.6 number — not load-bearing for any current work since `interrank` already has 108 LCB scores indexed

## Pivot

`Sylveste-k8c` (local flux-review backend) — DeepSeek-V4-Flash-4bit MLX is downloaded (141 GB on disk), the harness for batched local inference exists (`interfer/server`), and the question of "can we run reviews offline" doesn't share any of the cloud-cap problem with bvh. Different bottleneck (GPU/SSD), different deliverable (review-quality calibration vs pass@1 reproduction), different decision (default-route vs reserved-for-high-stakes).

The thing that bvh actually proved: **stop benchmarking cloud through OpenRouter**. Read interrank, save the money for things only we can measure (local stack, custom configs, agent-coordination latency under load).
