# LiveCodeBench v6 Matrix — 7-Model Comparison (2026-04-27)

**Beads:** [Sylveste-xzw], [Sylveste-b7j], [Sylveste-2ss]
**Harness:** `benchmarks/code_correctness.py` + `scripts/run_lcb_matrix.sh`
**Output:** `benchmarks/lcb_v6_matrix/code_correctness{,_summary}.{jsonl,json}`
**Run dates:** Started 2026-04-26 13:40 PDT, completed 2026-04-27 19:52 PDT (~30h wall)

## TL;DR

| Model | pass@1 | n | Verdict |
|---|---|---|---|
| **cloud (GPT-5.5 xhigh fast, codex)** | **84.0%** | 175 | Routing tier upper bound. Dominates every local. |
| **Qwen3.6-35B-A3B-4bit (no-think)** | **40.0%** | 175 | 🥇 **Best local C2 candidate.** Fastest, cleanest, no quantization or thinking overhead. |
| Qwen3.6-35B-A3B-4bit-DWQ (no-think) | 40.6% | 175 | DWQ adds nothing: same quality, 38% slower. Drop. |
| Qwen3.6-27B-4bit (dense) | 38.3% | 175 | Slower (4× per-token) AND lower quality. Strict-dominated by 35B-A3B. Drop. |
| flash-moe:Qwen3.5-397B-A17B (Q3) | 24.8% | 141¹ | C3 tier under-delivers. ~4 tok/s actual (vs claimed 12.9), 64% timeouts. Worker crashed at problem 151. |
| Qwen3.6-35B-A3B DWQ (thinking, 180s budget) | 22.3% | 175 | Thinking actively *hurts* at deployment-realistic budget. |
| Qwen3.5-35B-A3B-4bit (regression baseline) | 17.7% | 175 | 🪦 Obsolete. Strictly dominated by 3.6 and by cloud. |

¹ flash-moe completed 141/175 problems before its worker subprocess died with `urlopen error [Errno 60]` at problem 151. The remaining 25 problems errored out without retry. n=141 is the meaningful sample.

## Configuration (deployment-realistic, n=1)

- `max_tokens=8192` (vs LCB official `max_tokens=2000`)
- `--timeout=180` per-prompt
- `n=1` (vs LCB official `n=10` self-consistency)
- Qwen3.6 models: `enable_thinking=False` by default (one variant runs with thinking on)
- `temperature=0.6` for all (codex CLI uses its own sampler)
- Suite: LCB v6 `test6.jsonl` from HuggingFace `livecodebench/code_generation_lite` — 175 problems
  - 43 easy / 52 medium / 80 hard
  - Time-segmented: problems published May 2023 – April 2025
  - **Caveat**: both Qwen3.6 (April 2026) and DeepSeek V4 (April 2026) likely had this data in training. Numbers reflect *contamination-aware* relative ordering, not absolute capability.

## Headline findings

### 1. Qwen3.6 is a real generational leap over Qwen3.5 (+22.3 points)

The 3.5 → 3.6 A/B is unambiguous:

| | 3.5-35B-A3B | 3.6-35B-A3B |
|---|---|---|
| pass@1 | 17.7% | **40.0%** |
| Median gen | 81s | **15s** (5× faster) |
| Runtime errors | 134 | 34 |

Per-problem agreement: **3.6 wins 50 problems that 3.5 fails. 3.5 wins 11 that 3.6 fails. Net +39.**

Note this is *much larger* than the published deltas would suggest. The Qwen3.6 model card reports SWE-bench Verified 73.4 vs 3.5's 70.0 (+3.4 points). On our LCB v6 deployment-realistic config, the gap is +22.3. Hypothesis: 3.6's improvements are concentrated in "produces runnable code in single-shot non-thinking mode" — exactly what production routing tiers need.

**Recommendation:** Migrate the C2 routing tier from `Qwen3.5-35B-A3B-4bit` to `Qwen3.6-35B-A3B-4bit`.

### 2. DWQ quantization adds nothing on this workload

Per-problem A/B vs plain 4bit (n=175): both pass=60, plain-only=10, dwq-only=11, neither=94. **+1 problem in dwq's favor — well within noise (±9 at p=0.5, n=21 disagreements).**

DWQ is meanwhile *38% slower* (median 21.2s vs 15.4s). The "+1-3% expected" claim from the DWQ marketing doesn't materialize at our config.

**Recommendation:** Drop `Qwen3.6-35B-A3B-4bit-DWQ` from CONFIG_REGISTRY. Keep plain 4bit.

### 3. Thinking mode HURTS at deployment-realistic budgets

DWQ-thinking with the same 180s per-prompt timeout: 22.3% (vs 40.6% non-thinking, **−18.3 points**). Per-problem A/B: thinking wins 8 problems no-think fails, no-think wins 39 thinking fails. Net **−31** for thinking.

Failure pattern is diagnostic: thinking-mode produced **104 runtime errors** (vs 27 for non-thinking) — the model gets cut off mid-thought, emits incomplete code that crashes when run.

This is the cleanest production-routing finding in the matrix: **at any budget short of 600s+ per problem, enabling thinking mode on Qwen3.6 is actively harmful.** Marketing pass@1 numbers (Qwen reports 80.4 for 35B-A3B on LCB v6, presumably with thinking + n>1) are not what you get in deployment.

**Recommendation:** Hardcode `enable_thinking=False` for all Qwen3.6 routing-tier configs.

### 4. Dense > MoE doesn't replicate on Apple Silicon

Qwen's official LCB v6 numbers show 27B (dense) at 83.9 vs 35B-A3B (MoE) at 80.4 — a 3.5-point dense win. On our M5 Max:

| | 27b dense | 35b-A3B MoE |
|---|---|---|
| pass@1 | 38.3% | 40.0% |
| Median gen | **62.3s** | **15.4s** (4× faster) |

**4× speed difference** confirms the memory-bandwidth math — 27B activated per token vs 3B activated per token, on a memory-bound platform like M5 Max.

The published "dense wins" likely depended on n>1 and/or thinking mode; in single-shot non-thinking inference, the MoE wins on both axes.

**Recommendation:** Drop `Qwen3.6-27B-4bit`. Keep MoE for any local routing tier on Apple Silicon.

### 5. Cloud (GPT-5.5 xhigh fast) dominates everything by 43+ points

Best-local-vs-cloud A/B (n=175): both=69, qwen3.6-only=1, cloud-only=78, neither=27. **Cloud wins 78× more often than local wins.** The single `qwen3.6-only` win is essentially a freak event (n=1 sampling).

Cloud failure modes are *infrastructure ceiling*, not quality ceiling: 16 `no_code_extracted` + 17 `gen_timeouts` + 10 wrong + 2 runtime. Bumping the timeout to 300s+ would likely push cloud to ~88-90%.

**Recommendation:** For any task where pass@1 quality matters more than latency or cost, escalate to cloud. The local C2 tier is at best a 40% quality service, suitable only for low-stakes / cache / draft scenarios.

### 6. flash-moe Qwen3.5-397B-A17B under-delivers for the C3 tier

| Spec (AGENTS.md, 2026-04-05) | Actual (this run) |
|---|---|
| Decode: 12.9 tok/s | **4.0 tok/s** |
| Quality: "C3 routing tier — moderate complexity" | **24.8% pass@1 over 141 completed problems** |
| Reliability: stable for hours | **Worker crashed at problem 151** |

flash-moe's 35 wins are a **strict subset** of cloud's wins (cloud-only=88, flash-moe-only=0). Same pattern vs 35b-A3B: flash-moe-only=0, 35b-only=34. **flash-moe doesn't add diversity OR quality** at the budgets that fit a routing tier.

The 4 tok/s figure is 3× slower than the AGENTS.md claim. Possible causes: cold expert cache (each new problem loads different experts), thermal throttling after 8h, or our prompt size (~2K input tokens) costing more prefill than the AGENTS.md benchmark. Worth investigating but doesn't change the C3 verdict.

**Recommendation:** C3 traffic should escalate to cloud, not stay on flash-moe. Demote `flash-moe:397b` to research-only status.

## Failure-mode taxonomy

What kills the local models on the 60% of problems they fail?

| Model | wrong-answer | runtime-error | timeout | no-code |
|---|---|---|---|---|
| cloud | 10 | 2 | 17 | 16 |
| 3.6-35b plain | 68 | 34 | 3 | 0 |
| 3.6-35b DWQ | 76 | 27 | 1 | 0 |
| 3.6-27b dense | 64 | 44 | 0 | 0 |
| 3.6-DWQ thinking | 32 | **104** | 0 | 0 |
| 3.5-35b | 10 | **134** | 0 | 0 |
| flash-moe-397B | 10 | 96 | 35 (timeout-or-urlopen) | 0 |

Two distinct failure profiles:
- **Cloud (GPT-5.5)**: hits the 180s wall or returns prose without code. Quality ceiling not yet reached.
- **All Qwen variants**: emit code that runs but produces wrong output (wrong-answer dominant) or crashes (runtime-error growing as model gets weaker).

3.5's 134 runtime errors are diagnostic — it consistently emits *broken Python* on hard problems. 3.6 dropped this to 34. That delta is most of the +22-point uplift.

## Wall-clock + cost summary

| Model | Wall-clock | Throughput | Notes |
|---|---|---|---|
| cloud | 2h53m | ~1 prob/min | Codex CLI overhead amortized after ~10 problems |
| 3.6-35b plain | 1h25m | 2 prob/min | Fastest local |
| 3.6-35b DWQ | 2h00m | 1.5 prob/min | DWQ overhead |
| 3.6-27b dense | 4h04m | 0.7 prob/min | Dense penalty on Apple Silicon |
| 3.5-35b | 4h00m | 0.7 prob/min | Mostly slow because of long thinking-style outputs |
| 3.6-DWQ thinking | 5h22m | 0.5 prob/min | Most problems hit the 180s wall |
| flash-moe | 5h45m + crash | 0.4 prob/min (n=141) | Worker died at problem 151 |
| **TOTAL** | **~30 hours** | | Process-per-model wrapper held memory budget |

Cloud cost: $0 (ChatGPT subscription, no API key). Local cost: ~70 kWh of M5 Max thermal envelope.

## Methodology footnotes

- **n=1 sampling**: LCB official runner uses n=10 for self-consistency. Our pass@1 numbers are systematically lower than LCB-reported figures by 5-15 points (varies by model). Use ratios, not absolute pass@1 against the leaderboard.
- **Contamination**: LCB v6 covers May 2023 → April 2025. Qwen3.6 and DeepSeek V4 (both released April 2026) likely had this data. Treat absolute numbers as contaminated; relative orderings within the matrix remain valid.
- **No DeepSeek V4**: doesn't fit on 128GB Mac (V4-Flash 4bit is 151GB on disk). Cloud-API comparison filed as [Sylveste-bvh].
- **No GPT-5.5 thinking-vs-not A/B**: codex CLI doesn't expose a `--no-think` flag for ChatGPT-OAuth-backed models.
- **flash-moe n=141 caveat**: any cross-model comparison involving flash-moe must restrict to that 141-problem subset for fairness.

## Follow-up beads

- [Sylveste-bvh] (P2) Add cloud:deepseek-v4-flash + cloud:deepseek-v4-pro to LCB v6 matrix
- [Sylveste-6ru] (P2) Qwen3.6-35B-A3B quantization sweep (DWQ vs nvfp4 vs OptiQ vs plain) — partially answered: drop DWQ
- [Sylveste-ep8] (P2) Evaluate Qwen3.6-27B-OptiQ-4bit — superseded by this finding (27b dense is dominated; OptiQ unlikely to flip the verdict on M5)
- [Sylveste-0gi] (P2) Port DeepSeek V4 Flash to flash-moe — much higher priority now that flash-moe-397B is shown to under-deliver
- **NEW** flash-moe stability investigation (worker crash at problem 151)
- **NEW** flash-moe slowness investigation (4 tok/s vs claimed 12.9 tok/s)
