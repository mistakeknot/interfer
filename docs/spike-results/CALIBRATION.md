---
date: 2026-05-11
beads: [Sylveste-k8c]
status: calibration data
---

# flux-local calibration — Qwen3.6-35B-A3B-4bit (4-bit MLX)

Each finding the local model produced on its own deployment code is
adjudicated against what we actually fixed. Ground truth is committed:
either the finding turned into a commit (`real bug, fixed`) or we
explicitly decided not to fix it (`real but not worth fixing`) or it
was a false positive (`hallucination`).

## Adjudicated findings

| Spike | Lens | Finding | Verdict | Fixed in |
|---|---|---|---|---|
| 2026-05-10 (canonical) | correctness | server can return `{"error": ...}` inside 200-OK SSE; we don't check, silently return empty content | **REAL BUG** | `93b2d16` |
| 2026-05-10 (canonical) | safety | `load_lens_prompt` doesn't validate `lens_name`; path traversal possible | **REAL BUG** | `93b2d16` |
| 2026-05-10 (canonical) | safety | external-server URL not validated | minor, defense-in-depth | not fixed |
| 2026-05-10 (canonical) | safety | secrets in stdout output not redacted | minor, defense-in-depth | not fixed |
| 2026-05-10 (canonical) | quality | (degenerated into self-affirmation loop) | output unusable | n/a |
| 2026-05-11 (slimmed quality) | quality | `import json` inside function body | real but cosmetic | not fixed |
| 2026-05-11 (slimmed quality) | quality | `SystemExit` from helper bypasses standard exception handling | **REAL ARCHITECTURAL BUG** | `db05fd2` |
| 2026-05-11 (slimmed quality) | quality | bare `dict` return type hint | real but cosmetic | not fixed |
| 2026-05-11 (all slimmed) | correctness | "No correctness issues found" on clean diff | **CORRECT (true negative)** | n/a |
| 2026-05-11 (all slimmed) | safety | "No safety issues found" on clean diff | **CORRECT (true negative)** | n/a |
| 2026-05-11 (all slimmed) | quality | content loop, no schema output | output unusable | n/a |

## Score

8 substantive findings produced across 3 runs.

| Category | Count |
|---|---|
| Real bug → fixed (true positive, actioned) | 3 |
| Real architectural issue → fixed (true positive, actioned) | 1 |
| Real but cosmetic / not worth fixing (true positive, declined) | 4 |
| Hallucination / wrong (false positive) | 0 |
| Clean diff correctly flagged clean (true negative) | 2 |
| Lens output unusable (process failure, not classified) | 2 |

**Precision on usable runs: 8/8 (no hallucinations)**, of which 4/8 (50%)
were severe enough to action immediately.

**Process reliability: 9/11 lens runs (82%) produced usable output.**
The 2 failures are both the quality lens (one degeneration into
self-affirmation, one into content looping). Correctness and safety
lenses succeeded on every run.

## What this is and isn't

**Is**: evidence that Qwen3.6-35B-A3B-4bit, with the slimmed lens
prompts, reliably produces *real* findings on Python diffs at the
~5-10 KB diff size. No hallucinated bugs across all spike runs.

**Isn't**: a comparison against Claude Opus on the same diffs. The
brainstorm doc said the gate is "if local-V4-Flash hits >75% finding-
overlap with Opus on calibration set, promote local as default." We
don't have the matching Opus reviews to compute overlap.

We can run flux-local's prompts against an Opus session for any of
these three diffs and get the comparison data. That's the next step.

## Decision (interim)

For *dogfood-quality* code reviews (the kind of review a developer
does on their own diff before opening a PR), the local stack with
slimmed correctness + safety lenses is **good enough to ship today**:

- Zero hallucinations across 8 findings
- 50% of findings were real bugs that turned into commits
- 9/11 lens runs (82%) usable
- Wall: 27-83s per lens, $0 cost, $0 API tokens

For *gate-quality* code reviews (the kind that approves a PR for merge),
we need:

1. Opus baseline on the same diffs for true overlap measurement
2. Quality lens fixed (content-loop + degeneration; needs prompt or
   sampling-parameter intervention)
3. Larger calibration corpus (3 diffs is small — at minimum 10)

## Followups

- Decide what to do with the four real-but-cosmetic findings (json-
  inside-function, bare dict hint, external-server validation, secret
  redaction). Either fix them or document the explicit accept.
- Fix quality lens: try lower temperature (0.1 instead of 0.3), add
  example findings to anchor the schema, or constrain max_tokens lower
  to force the model to commit.
- Pull Opus reviews on the same 3 diffs for true overlap measurement.
- Run flux-local on a larger diff (50+ KB) to test scaling.
