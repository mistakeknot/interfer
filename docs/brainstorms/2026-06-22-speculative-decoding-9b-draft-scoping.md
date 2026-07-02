---
artifact_type: scoping
bead: sylveste-yfot
stage: scope
date: 2026-06-22
status: draft
---

# Scoping: Benchmark speculative decoding with 9B draft model (sylveste-yfot)

**Bead:** sylveste-yfot — "[interfer] Benchmark speculative decoding with 9B draft model"
**Original framing:** "Standard speculative decoding using Qwen3.5-9B as draft for 35B-A3B
target. Already wired in inference.py via draft_model_name. Benchmark acceptance rate,
tok/s speedup, quality."

This is a SCOPING note, not an experiment run. Recommendation up front, then evidence.

## Recommendation: **likely-moot** (but a cheap Phase-1 measurement should confirm)

The proposed draft/target pairing inverts the core economic premise of speculative
decoding. Spec-decode wins only when the draft model's per-token cost is *much smaller*
than the target's per-token cost. Here:

- **Target** = `qwen3.5-35b-a3b-4bit` — a **MoE with only 3B active params**,
  benchmarked at **~86 tok/s** (`interfer/AGENTS.md:62`).
- **Draft** = `qwen3.5-9b-4bit` — a **dense 9B**, benchmarked at **~60-80 tok/s**
  (`interfer/AGENTS.md:61`).

The draft is *slower or comparable* per token to the target, because the target only
pays for 3B of active compute while the draft pays for all 9B. The classic spec-decode
win (e.g. DeepSeek/Llama: tiny dense draft for a huge dense target) does not map onto a
3B-active MoE target. The draft has to be cheaper than the target's *active* cost, and a
9B dense model is not.

This deserves a Phase-1 measurement (not a full sweep) precisely because it is cheap to
run and would close the question definitively, per the platform's
test-null-hypothesis-first doctrine.

## Verified claims (against code, not memory)

- Spec decoding IS wired:
  - `server/inference.py:223-224` — `draft_model_name: str | None`, `num_draft_tokens: int = 3`
  - `server/inference.py:353-357` — loads draft via `_ensure_loaded`, sets `gen_kwargs["num_draft_tokens"]`
  - `server/inference.py:398` — passes `draft_model=draft_model_obj` into `stream_generate`
  - `server/benchmark.py:139-140, 195-196, 221-222, 288-292` — benchmark harness exposes draft + reports it
  - `server/benchmark_cli.py:70-76, 103-104, 121` — CLI flags `--draft-model`, `--num-draft-tokens`
  - The bead's "already wired in inference.py via draft_model_name" claim is **accurate**.
- Target is MoE 3B-active at ~86 tok/s; draft is dense 9B at ~60-80 tok/s
  (`interfer/AGENTS.md:61-62, 65-66`).
- A **sibling experiment already exists**: LayerSkip self-speculative decoding (sylveste-qbv,
  `docs/reflections/2026-04-07-layerskip-poc.md`) got **0% acceptance across 16 configs** on
  Qwen3.5 MoE and concluded MoE expert routing "distributes computation … there's no early
  completion point." It explicitly named "standard speculative decoding with a separate
  draft model (already supported via `draft_model_name`)" as the untested fallback — i.e.
  sylveste-yfot is the never-run follow-up to a sibling that already failed for
  MoE-architectural reasons.

## Why skeptical (overlap + weak evidence base)

1. **Architecture mismatch.** Spec-decode math: speedup ≈ (accepted_tokens_per_step) /
   (1 + draft_cost_ratio). With draft_cost_ratio ≈ 1 (9B dense ≈ 3B-active MoE in
   wall-clock per token), even a high acceptance rate yields little net speedup, and a
   mediocre acceptance rate yields a **slowdown** (you pay draft cost for rejected tokens
   on top of full target verification).
2. **Vocabulary/tokenizer coupling.** mlx-lm `stream_generate` spec-decode requires the
   draft and target to share a tokenizer/vocab. Both being Qwen3.5 is favorable here — but
   this is an assumption to verify in Phase-1, not assume.
3. **Acceptance is the unknown, and a sibling already says it's bad for this family.**
   LayerSkip's 0% is self-speculative (same weights, early exit), which is a different
   mechanism than a separate draft model — so it is NOT conclusive for separate-draft
   spec-decode. But it is a strong prior that Qwen3.5 MoE token distributions are not
   easily predicted by a cheaper proxy.
4. **C2 is already fast enough for its tier.** The C2 routine tier runs at ~86 tok/s
   locally. There is no recorded user-facing latency pain at C2 (no bead, no handoff cites
   it). Optimizing a tier that already meets its SLA is low-value even if it worked.

## Testable hypothesis

Using `qwen3.5-9b-4bit` as a draft for the `qwen3.5-35b-a3b-4bit` target via the existing
`draft_model_name` path yields a **net decode-throughput speedup of ≥1.3x** on the
interfer code-generation prompt set at default `num_draft_tokens=3`, with **output quality
identical** (greedy, temperature 0 → token-for-token identical by spec-decode's
correctness guarantee).

## Pre-registered KILL RULE (Phase-1, half-day)

Run a single A/B on the existing harness:
`benchmark_cli.py --model local:qwen3.5-35b-a3b-4bit` with and without
`--draft-model local:qwen3.5-9b-4bit`, over the existing benchmark prompt set, temperature 0.

- **KILL (close sylveste-yfot, do not file followups)** if EITHER:
  - measured net tok/s with draft is **< 1.15x** the no-draft baseline (i.e. < 15%
    speedup — below the threshold where it justifies holding a 5GB draft resident and the
    coexistence/RAM cost), OR
  - measured **acceptance rate < 50%** at `num_draft_tokens=3` (confirms the MoE
    target-distribution-not-predictable prior).
- **NARROW (one Phase-2 bead)** if speedup is 1.15x–1.30x: sweep `num_draft_tokens ∈
  {2,4,5}` and try a smaller draft (e.g. a 1.5B/3B dense Qwen3.5) to find the real
  cost-optimal draft, since the 9B is almost certainly too large a draft.
- **PURSUE** only if ≥1.30x with ≥50% acceptance: then wire a routing-tier default and
  measure RAM-coexistence impact (9B draft + 35B target + KV pool simultaneously resident).

Quality is not a kill axis at temperature 0 — spec-decode is output-exact by construction;
the Phase-1 run should *assert* token-identity as a correctness check, and any divergence
is a harness bug to fix, not an experiment result.

## Method in brief

1. Confirm both models cached/loadable and share tokenizer/vocab (5 min).
2. Baseline: `benchmark_cli.py --model local:qwen3.5-35b-a3b-4bit` over the existing
   prompt set, temp 0, capture mean tok/s.
3. Spec: same command + `--draft-model local:qwen3.5-9b-4bit --num-draft-tokens 3`,
   capture mean tok/s + acceptance rate (harness already reports draft metadata at
   `benchmark.py:288-292`; confirm acceptance rate is surfaced — if not, that is a
   one-line addition to the summary, not new infra).
4. Assert token-identity baseline-vs-spec at temp 0.
5. Apply kill rule above. Result note → `docs/reflections/2026-06-XX-spec-decode-9b-draft.md`.

## Rough effort

- Phase-1 measurement: **hours** (half-day). All infra exists; this is a benchmark run +
  a kill-rule decision, not a build.
- Full pursue path (only if Phase-1 passes): days (draft-size sweep + RAM-coexistence work).

## Honest take

This reads as a leftover experiment idea from the early "memory budget" planning
(`docs/interfer-prd.md:78, 109` reserve "~5GB draft (8B Q4)" and target "1.8x speedup
with 65%+ acceptance") that predates the MoE-first pivot. Once the C2 target became a
3B-active MoE running at 86 tok/s, the rationale for a 9B dense draft largely evaporated —
the draft is no longer "the cheap one." The sibling LayerSkip failure reinforces that
Qwen3.5 MoE token streams resist cheap prediction. I expect Phase-1 to hit the kill rule.
But because the run is half a day and would *settle* the question (and the harness already
supports it), running Phase-1 once and closing on the kill rule is cleaner than parking it
to resurface again. **likely-moot, confirm-then-close.**
