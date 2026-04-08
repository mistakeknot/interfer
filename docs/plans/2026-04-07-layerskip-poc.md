---
artifact_type: plan
bead: sylveste-qbv
stage: planned
---

# Plan: LayerSkip Self-Speculative Early Exit — Proof of Concept

**Bead:** sylveste-qbv
**Date:** 2026-04-07

## Task 1: Implement `self_speculative_generate` in early_exit.py

**File:** `server/experiments/early_exit.py`

Add a standalone function that demonstrates self-speculative decoding:

```python
def self_speculative_generate(
    model, tokenizer, prompt: str,
    exit_layer: int = 32,
    confidence_threshold: float = 0.95,
    max_tokens: int = 100,
) -> dict:
```

Logic:
1. Tokenize prompt, create KV cache
2. Embed tokens: `h = model.model.embed_tokens(tokens)`
3. Run prompt through ALL layers (prefill — no shortcut for context)
4. For each decode step:
   a. Run layers[0..exit_layer-1] on current token
   b. Apply `model.model.norm(h)` then `model.lm_head(normed_h)` → draft logits
   c. If `max(softmax(draft_logits)) > threshold`: accept draft token
   d. Else: run layers[exit_layer..end] → apply norm + lm_head → verified logits
5. Track: accepted_count, verified_count, total_tokens, elapsed time

Return dict with: tokens generated, acceptance rate, tok/s, text output.

**Key details:**
- Use proper masks: `create_attention_mask` for full-attn layers, `create_ssm_mask` for linear layers
- Reuse KV cache between draft and verify (layers 0..exit_layer already cached)
- Apply norm before lm_head (critical — hidden states aren't in projection space without it)

## Task 2: Write benchmark script

**File:** `benchmarks/layerskip_benchmark.py`

Run the PoC on 4 prompts from `prompts/coding_eval.json`:
- Compare: standard generate vs self_speculative_generate
- Measure: tok/s, acceptance rate, output quality (visual comparison)
- Sweep: exit_layer in [16, 24, 32] for 64-layer model

## Task 3: Run benchmark on Qwen3.5-35B-A3B

Load the 35B MoE model and run both standard and self-speculative.
Record results in TSV.

## Task 4: Write results doc

Analyze: does self-speculative give >1.2x speedup? Is quality preserved?
Decision: proceed to full integration or close as not-viable.

## Execution Order

Task 1 → Task 2 → Task 3 → Task 4 (sequential — each depends on prior)
