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
