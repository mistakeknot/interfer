---
date: 2026-05-11
session: b199f064
topic: k8c flux-local ship + cosmetic findings + quality lens + Opus A/B
beads: [Sylveste-k8c, Sylveste-bvh, Sylveste-bov, Sylveste-uk3]
---

## Session Handoff — 2026-05-11 k8c flux-local — G/H/I done, J only

### Directive

> Your job is J: Opus A/B vs the local-Qwen reviews. G+H+I already shipped and pushed in this session. **Do NOT boot a fresh interfer server** for J — last run leaked Metal-worker processes that bloated Activity Monitor's "real memory" column to look like ~138 GB consumed (it was mmap accounting, not distinct RAM, but the orphans are real). All 3 diffs already on disk; reuse them via Agent calls to Opus.

**J — parallel Opus A/B (1 hr):**
Spawn 3 parallel `Agent` calls (default `general-purpose` = Opus 4.7), one per diff. Each agent gets:
1. The diff at `/tmp/flux-local-{3f60481,5dad6cf,db05fd2}.diff` (still on disk; regenerate via `env -u GIT_INDEX_FILE git show <hash> > /tmp/flux-local-<hash>.diff` from `interverse/interfer/` if cleared)
2. All three **canonical** lens prompts from `~/.claude/plugins/cache/interagency-marketplace/interflux/0.2.69/agents/review/fd-{correctness,safety,quality}.md`
3. Instruction to return findings in the same SEVERITY:file:line schema the local model uses

Score finding-overlap against `docs/spike-results/CALIBRATION.md`. Update CALIBRATION.md § Decision based on overlap. If >75%, promote local-default per brainstorm gate; if not, document the gap.

**No new beads needed** — Sylveste-k8c (in_progress) covers this; just update its notes when J lands.

### Dead Ends

- **OpenRouter via Novita/DeepInfra for V4 Flash** — silently caps `max_tokens` at 16384. Probe abc398_g: asked 64000 → got 16384, `finish_reason='length'`, 16607 reasoning tokens, 0 text. Cloud LCB → use `interrank`. Don't reopen unless using native `api.deepseek.com`.
- **`hf` CLI in homebrew Python 3.11** — missing `certifi`, ignores `uv run --with`. Use `huggingface_hub.snapshot_download` from project venv.
- **`stdbuf` for nohup flushing** — Homebrew coreutils libstdbuf.so is x86_64; DYLD rejects in arm64 Python. Use `PYTHONUNBUFFERED=1`.
- **mlx-lm 0.31.1 + DeepSeek V4 Flash** — `deepseek_v4` arch NOT in mlx-lm registry. Model on disk at `/Users/sma/Models/DeepSeek-V4-Flash-4bit-mlx/` (141 GB, 33 shards) but unloadable until mlx-lm adds handler. Used Qwen3.6-35B-A3B-4bit instead.
- **Per-process model loading** — won't parallelize on M5 Max; GPU serializes Metal contexts. Use one server + N concurrent requests.
- **NEW: interfer server orphans on `pkill`** — `pkill -f "python -m server"` kills the parent `uv run` and Starlette, but the spawned `multiprocessing` Metal-worker subprocess survives (separate process group) and keeps mmapped weights resident. Each subsequent server boot adds another ~18 GB of "real memory" in Activity Monitor (mmap accounting per process — physical pages are shared, so it's not 18 GB of *new* RAM, but reading the screenshot misleads). **Clean kill: `pkill -f "python -m server"` then also `pgrep -f "spawn_main" | xargs kill`**, or just kill the `uv run` parent pid and wait — its grandchildren reap when the kernel notices.

### Context

- **All 13 commits pushed** to `interfer/main`. `git log origin/main..HEAD` should be empty as of session close.
- **2 commits pushed this session** after the handoff was first written: `a0dac81` (H — 4 cosmetic findings closed) and `8811792` (I — quality lens commits, no more loop). Confirmed via `git push origin main` → `9141d7b..8811792`.
- **H findings closed**: top-level `json` import; `LensResult(TypedDict)` replaces bare `dict` return; warn-on-non-loopback `--server`; `_redact_secrets()` covers `sk-or-v1-*`, `sk-ant-*`, `sk-proj-*`, `ghp_*`, `github_pat_*`, `AKIA*`.
- **I fix**: `temperature=0.3` → `0.1` in `scripts/flux-local.py:run_lens` + added 2 example findings at the bottom of `scripts/lens-local/fd-quality.md` ("Example findings (use this exact shape)" section). Re-run on `db05fd2`: 33s wall, emits exactly `No quality issues found`. Spike result in `docs/spike-results/2026-05-11-flux-local-db05fd2-quality-tuned.txt`.
- **OPENROUTER_API_KEY** at `/Users/sma/.cache/interfer/openrouter.key` (mode 600, not in repo). User pasted the key in the 2026-05-08 transcript; rotate at openrouter.ai/keys before relying on it again.
- **Calibration ground truth**: 8 substantive findings from 3 spike runs, 0 hallucinations, 4 became commits, 2 true negatives, 2 quality-lens process failures (1 self-affirmation loop, 1 content loop — latter fixed by I). Adjudication table in `docs/spike-results/CALIBRATION.md`. **Use as baseline for J's overlap measurement.**
- **interfer server `/v1/chat/completions`** at `interverse/interfer/server/main.py:1040` always streams SSE (no JSON branch). flux-local handles SSE accumulation client-side. Default port 8421; spike runs used 8423.
- **Hook collision**: `security_reminder_hook.py` pattern-matches specific function names (binary-deserialize, system-shell-exec) — fires even in *warnings against* using them. Rephrase as "unsafe deserialization" / "system-shell calls" if it blocks a write.
- **Subrepo trap**: `interverse/interfer/` is its own git repo. Outer monorepo `.gitignore` excludes it. All k8c commits/pushes from `/Users/sma/projects/Sylveste/interverse/interfer/`. Always `env -u GIT_INDEX_FILE` prefix.
- **Key paths for J**:
  - Canonical lenses (use these for Opus side): `~/.claude/plugins/cache/interagency-marketplace/interflux/0.2.69/agents/review/fd-{correctness,safety,quality}.md`
  - Slimmed lenses (already used on local side): `interverse/interfer/scripts/lens-local/fd-{correctness,safety,quality}.md`
  - Adjudicated findings: `interverse/interfer/docs/spike-results/CALIBRATION.md`
  - Diff hashes for A/B: `3f60481`, `5dad6cf`, `db05fd2` (capture via `env -u GIT_INDEX_FILE git show <hash>` from interfer subrepo)
- **bd push.sh requires tty** — `bash .beads/push.sh` gates on interactive confirm. Tell the user to run it; don't bypass.
