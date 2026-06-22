---
bead: sylveste-c57
date: 2026-06-22
type: scoping-spike
status: SCOPED (not started)
recommend: likely-moot
---

# Scoping: Ant Colony Optimization for Expert Routing Paths (sylveste-c57)

## The bead's claim

> Track which expert sequences produce high-confidence outputs (pheromone trails).
> Over many requests, expert routing paths that frequently produce good results get
> reinforced, while poor paths decay. This creates an emergent 'highway system'
> through the expert space. Could dramatically improve page cache hit rates for SSD
> streaming — if the same expert sequences are preferred, they stay warm in cache.

Tagged "research-grade, high risk/high reward."

## Why I am skeptical (verified against code, not memory)

### 1. The stated bottleneck is already known NOT to be the bottleneck

`docs/benchmarks/streaming_phase0_analysis.md` (bead sylveste-14g, 2026-03-28) measured
the SSD-streaming decode path directly:

- pread I/O: 68% of per-token time **but** "the bottleneck is NOT I/O bandwidth — it's
  the 60 per-layer CPU-GPU synchronization points" (line 21-22).
- NVMe measured at 47.5 GB/s parallel / 14.5 GB/s per-thread libc pread (line 94).

ACO improves *page-cache hit rate* — i.e., it reduces pread volume. But pread bandwidth
is not the constraint; per-layer GPU sync and serial layer execution are. Reducing cache
misses on an already-fast I/O path cannot "dramatically" move throughput. The bead
attacks the wrong term in the latency budget.

### 2. A Zipf-aware LRU already captures almost all the available cache benefit

`docs/reflections/2026-03-29-flashmoe-cache-tuning.md` (bead sylveste-vm4, line 18):

> Expert routing follows Zipf distribution — a cache holding 8% of experts can achieve
> ~60% hit rate. This means the first few thousand entries have outsized impact;
> diminishing returns past ~10,000.

The hot experts are *already* kept warm by the existing `--malloc-cache` LRU (GPU-resident,
zero-copy Metal buffers). ACO is a complicated mechanism whose entire purpose is to keep
hot paths warm — which a plain frequency-ordered cache does for free under a Zipf
distribution. There is little residual hit-rate headroom for a smarter policy to claim,
and the residual is in the cold long tail where reinforcement helps least.

### 3. The legitimate version of this idea already exists and is unfinished

The binary already ships a `--predict` flag: "a temporal predictor trained on routing
data" (cache-tuning reflection, line 23; wired in `server/flashmoe_worker.py:69,133-134`
as `predict: bool`). The honest, evidence-based form of "warm the experts we are about to
use" is *next-layer expert prediction* — already named as Optimization Path #2 in the
phase0 analysis (line 63): "Use layer N's routing to predict layer N+1's likely experts."
ACO is a swarm-intelligence reframing of a predictor that already exists in skeleton form
and has not even been evaluated yet (the reflection's "for next time" notes `--predict`
"needs separate evaluation").

If anyone wants to spend effort here, the correct move is to **evaluate the existing
`--predict` predictor**, not to build an ACO layer on top of it.

### 4. The control surface is in a third-party C/Metal binary, not in interfer

The shipped architecture (phase0 "Decision", line 81-84) is **Option C**: flash-moe
(Anemll `m5-nax` branch, Objective-C + Metal, `~/projects/flash-moe/metal_infer/`) is the
inference backend; interfer is only an HTTP proxy (`server/flashmoe_worker.py` spawns the
binary and forwards SSE). Expert selection (router top-k), the expert cache, and eviction
all live inside `main.m` / `infer.m` / `shaders.metal`. interfer's only knobs are CLI
flags (`--malloc-cache`, `--cache-io-split`, `--q3-experts`, `--predict`).

Implementing pheromone-weighted routing means modifying an upstream fork's Metal pipeline
to (a) bias top-k selection by an accumulated pheromone table and (b) carry that state
across requests. That is deep surgery in a vendored C/Metal codebase we track at a pinned
commit — high cost, high merge-maintenance burden, and squarely outside interfer's
"HTTP server + experiment hooks" scope.

### 5. The objective is internally conflated

ACO reinforces paths by **output confidence** ("high-confidence outputs"). The page-cache
benefit it claims requires reinforcing **frequently-used** experts. These are different
objectives: you cache what is *hot*, not what is *confident*. Biasing routing toward
high-confidence historical paths is also a correctness hazard — it changes which experts
the model actually selects, i.e., it perturbs model outputs to chase a cache metric. That
trades quality for an I/O win on a path where I/O is not the bottleneck (see #1).

## Hypothesis (testable, if anyone insists)

> Reordering/biasing expert selection toward historically-reinforced "trails" raises the
> flash-moe expert-cache hit rate by >15 percentage points over the existing Zipf-LRU
> `--malloc-cache` baseline, AND that hit-rate gain converts to >15% decode tok/s on the
> 397B SSD-streamed path, WITHOUT degrading PPL by more than 2%.

All three conjuncts must hold for the idea to be worth building. Given #1 (I/O is not the
bottleneck) and #5 (routing changes perturb quality), the joint probability is low.

## Pre-registered KILL RULE (Phase-1 measurement first, per platform doctrine)

Before writing any ACO code, run a **null-hypothesis measurement** using tooling that
already exists — no binary changes required:

**Phase 1 (effort: hours).** Use `benchmarks/cache_sweep.py` (already parses
`hit_rate_pct` and `expert_io_pct`, see lines 78, 135-138, 365-371) plus a one-off
routing-trace dump (the `--predict` path implies `--collect-routing` exists upstream;
if not, log router top-k indices per layer per token to a file from the binary's existing
stderr stream). Measure two things on the recommended Q3 config:

1. **Cross-token expert-set Jaccard overlap** at decode (how stable are the active
   experts token-to-token, layer-by-layer). ACO only helps if trails are *stable*.
2. **Marginal hit-rate headroom**: current `--malloc-cache 5000` hit rate vs. an oracle
   upper bound (Bélády / perfect-knowledge cache replay over the trace).

**KILL conditions (close sylveste-c57 MOOT, file no followups) if ANY hold:**

- Existing malloc-cache hit rate is already within **10 percentage points** of the Bélády
  oracle upper bound. (No room for a smarter policy to matter.)
- Cross-token expert-set Jaccard overlap is **< 0.3** at decode. (No stable "trails"
  exist to reinforce; the Zipf hotset is the only structure, and LRU already exploits it.)
- The cache-tuning Zipf finding holds and decode tok/s on the recommended config is
  **I/O-bound by < 20% of per-token time** when re-profiled. (Removing I/O misses can't
  move the headline number.)

**PROCEED to a Phase-2 spike only if** all three: oracle headroom > 10pp, Jaccard > 0.3,
AND I/O is > 20% of the per-token budget. Even then, the first Phase-2 action is to
evaluate the existing `--predict` temporal predictor — NOT to build ACO. ACO is only
justified if `--predict` is measured and leaves > 10pp of the oracle headroom unclaimed.

## Method in brief

1. (hours) Dump a decode routing trace from flash-moe on a representative prompt set.
2. (hours) Compute cross-token Jaccard overlap + Bélády oracle hit rate vs. current
   malloc-cache hit rate; re-profile the I/O fraction of per-token time.
3. Apply the kill rule. Expected outcome: MOOT.
4. (only if survives) Evaluate `--predict` against the malloc-cache baseline.
5. (only if `--predict` leaves headroom) Scope ACO as an upstream flash-moe fork change.

## Rough effort

- Phase-1 measurement: **hours** (existing benchmark + a trace dump).
- Phase-2 (if it survives): days to evaluate `--predict`; **weeks** for any actual ACO
  implementation, because it requires modifying the vendored C/Metal binary and carrying
  cross-request state — the highest-cost change surface in the interfer ecosystem.

## Recommendation: PARK (lean MOOT)

Do not start the ACO experiment. The bead's headline mechanism targets a bottleneck that
phase0 already disproved, claims a hit-rate win that a Zipf-LRU already captures, and
duplicates an existing `--predict` predictor that has never been evaluated — all while
requiring surgery in a third-party Metal binary.

If session time is ever spent in this area, spend it on the cheap, already-scoped wins:
(a) run the deferred `cache_sweep.py` sweep, and (b) evaluate the existing `--predict`
flag. Those are the legitimate descendants of this idea and they dominate ACO on
effort-adjusted expected value. The ACO framing itself is "interesting metaphor, weak
evidence base" — recommend MOOT after the hours-scale Phase-1 measurement, with the three
kill conditions above making MOOT the most likely outcome.
