# flux-local spike results

Captures from running `scripts/flux-local.py` against historical diffs for
Sylveste-k8c calibration. Each file is the verbatim output of one fan-out:
N lenses, one local model, one diff.

Use these to:
- Eyeball review quality on a real model + diff
- Spot lens prompts that degenerate on smaller models (e.g. quality lens
  recursive self-affirmation loop on Qwen3.6-35B-A3B at 8K max_tokens)
- Compare wall-clock and token usage across diffs

## Files

| File | Diff | Model | Lenses | Wall | Notes |
|---|---|---|---|---|---|
| `2026-05-10-flux-local-3f60481.txt` | flux-local.py initial commit (~10 KB diff) | Qwen3.6-35B-A3B-4bit | correctness, safety, quality | 6 min | correctness + safety produced real findings; quality lens degenerated into "I will be precise..." loop |
| `2026-05-11-flux-local-3f60481-slimmed-quality.txt` | flux-local.py initial commit (~10 KB diff) | Qwen3.6-35B-A3B-4bit | quality (slimmed) | 35s | Slimmed Python-only quality lens, explicit output schema, "no preamble" instruction. 3 real findings: import-inside-function, SystemExit-from-helper, bare dict type hint. **5× faster, ~2× fewer tokens, and actually useful** — validates brainstorm Step 3. |
| `2026-05-11-flux-local-db05fd2-all-slimmed.txt` | SystemExit→FluxLocalError fix (~3 KB diff) | Qwen3.6-35B-A3B-4bit | correctness, safety, quality — all slimmed | **83s for all 3** | **4.4× faster than original** (was 365s on a similar fan-out). Correctness and safety both correctly identified the clean diff and emitted "No \<lens\> issues found" per the slimmed prompt's exit instruction. Quality lens has a new failure mode: *content looping* — repeats the same 8-line analysis 3 times, never reaches the SEVERITY:file:line schema. Suggests quality lens needs (a) stronger clean-path reinforcement, or (b) lower temperature, or (c) example findings in the prompt to anchor the schema. |
| `2026-05-11-flux-local-db05fd2-quality-tuned.txt` | same SystemExit fix diff (~3 KB) | Qwen3.6-35B-A3B-4bit | quality — slim + 2 example findings + temp 0.1 | 33s | Quality-lens looping fixed. Two changes from the previous run: temperature lowered 0.3→0.1, and two example findings added at the bottom of the prompt to anchor the schema. Output is exactly `No quality issues found` (the clean-path string). Validates k8c step I. The "do not write your reasoning between findings" + "do not repeat findings" lines also help. |
