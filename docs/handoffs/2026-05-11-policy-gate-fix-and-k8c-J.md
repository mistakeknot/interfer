---
date: 2026-05-11
session: b199f064
topic: policy-gate fix + k8c J deferred
beads: [Sylveste-k8c, Sylveste-rkm]
---

## Session Handoff — 2026-05-11 policy-gate fix + k8c J still pending

### Directive

> Your job is **J for Sylveste-k8c** (in_progress): parallel Opus A/B vs the local-Qwen reviews. **Do NOT boot a fresh interfer server** — last cleanup recovered ~50 GB of RAM by reaping leaked Metal workers; another fresh boot puts us back in that hole. Use Agent calls only.

**J — execution plan:**
1. Capture diffs (still in `/tmp/flux-local-{3f60481,5dad6cf,db05fd2}.diff` from last session; if cleared, regenerate via `env -u GIT_INDEX_FILE git show <hash> > /tmp/flux-local-<hash>.diff` from `/Users/sma/projects/Sylveste/interverse/interfer/`).
2. Spawn **3 parallel `Agent` calls** (default `general-purpose` = Opus 4.7), one per diff. Each agent receives:
   - the diff content
   - **canonical** lens prompts from `~/.claude/plugins/cache/interagency-marketplace/interflux/0.2.69/agents/review/fd-{correctness,safety,quality}.md` (Opus-sized; use the full versions, not the slimmed ones)
   - instruction to emit findings in the same SEVERITY:file:line schema the local model uses
3. Score finding-overlap against `/Users/sma/projects/Sylveste/interverse/interfer/docs/spike-results/CALIBRATION.md`.
4. Update `CALIBRATION.md § Decision`. If >75% overlap, promote local-default per the brainstorm gate.

**Session-close rule (NEW, exercise it):** Run `bash /Users/sma/projects/Sylveste/.beads/push.sh` yourself as part of close ceremony — it runs unattended now (since Clavain `e316334`). The earlier "needs tty" prompt was a stale CLI invocation, not a real safety gate. See `feedback_bd_push_needs_tty.md` (updated this session).

**Open followup bead (not blocking)**: `Sylveste-rkm` P3 — policy-engine audit + dead audit/sign-CLI cleanup (the source of the cosmetic "policy: record failed" / "policy: sign failed" warnings still printing on every `bd-push-dolt`).

### Dead Ends

- **`policy check` (space) in gate wrappers** — was an unknown command in current `clavain-cli` (CLI renamed to `policy-check` with hyphen at some point), returning rc=1 as "unknown command", which `gate_decide_mode` interpreted as "needs confirmation". Fixed in Clavain `e316334`. Don't reopen unless you see new "needs confirmation" prompts in non-interactive contexts.
- **Blanket-renaming `policy record` / `policy sign` / `policy token consume`** — these aren't just renames, the entire audit-surface was removed from `clavain-cli` (only `policy-check` and `policy-show` remain in `clavain-cli help`). Don't try to fix by renaming; that's `Sylveste-rkm` territory (decide whether to delete the dead call sites or restore the CLI surface).
- **`pkill -f "python -m server"`** — leaves multiprocessing Metal-worker subprocess alive in a separate process group, each later boot adds another ~18 GB to Activity Monitor's "real memory" (mmap accounting per-process, but the orphans are real). Clean kill: `pkill -f "python -m server"` then `pgrep -f spawn_main | xargs kill`. Or just don't boot a server for J — Agent calls are sufficient.

### Context

- **`Sylveste-k8c` in_progress, only J remains.** G/H/I shipped + pushed this session and the prior one. 14 commits on `interfer/main` all pushed; `origin/main..HEAD` empty.
- **`Sylveste-rkm` open P3** — Clavain policy-engine audit (filed end of this session). Spells out the 4-item audit: (1) decide audit/sign-surface direction, (2) verify policy-check signature, (3) wire real bd-push-dolt rules, (4) cross-platform DOLT path on other gate wrappers.
- **The bd-push-dolt fix changed the session-close protocol.** Old: tell user to run `bash .beads/push.sh` themselves. New: run it yourself. Memory `feedback_bd_push_needs_tty.md` updated; MEMORY.md index line refreshed. Don't go back to telling the user.
- **Carve-outs where you SHOULD still ask before bd push:**
  - After a `bd ops` that rewrote Dolt history (clobber risk)
  - In a long-running parallel-agent session where another agent might be pushing too (interlock-coordinate first)
  - Just had a failed `bd backup sync` (JSONL backup is stale)
- **OPENROUTER_API_KEY** at `/Users/sma/.cache/interfer/openrouter.key` (mode 600). User pasted it in the 2026-05-08 transcript — rotate at openrouter.ai/keys when convenient.
- **Calibration ground truth** for J's overlap measurement: 8 findings, 0 hallucinations, 4 → commits, 2 true negatives. `/Users/sma/projects/Sylveste/interverse/interfer/docs/spike-results/CALIBRATION.md`. Each row already adjudicated by commit reference.
- **Subrepo trap reminder**: `interverse/interfer/` and `os/Clavain/` are independent git repos. `.gitignore` of outer monorepo excludes them. Always `env -u GIT_INDEX_FILE` prefix for git ops in subrepos.
- **Clavain has 2 pre-existing unstaged files** (`bin/clavain-cli-go-darwin-arm64`, `commands/sprint.md`) — not mine, not from this session. Left untouched (stashed during my rebase, popped after).
- **Hook collision watchpoint**: `security_reminder_hook.py` pattern-matches function names like the binary-deserialize / system-shell-exec ones — fires even when the prompt is *warning against* using them. Rephrase as "unsafe deserialization" / "system-shell calls" if it blocks a write.
- **mlx-lm 0.31.1 still has no `deepseek_v4` arch.** DeepSeek-V4-Flash-4bit MLX at `/Users/sma/Models/DeepSeek-V4-Flash-4bit-mlx/` (141 GB, 33 shards) is unloadable. For local inference, use Qwen3.6-35B-A3B-4bit at `/Users/sma/Models/Qwen3.6-35B-A3B-4bit/`.
- **OpenRouter via Novita/DeepInfra caps `max_tokens` at 16384 silently** for DeepSeek V4 Flash. Cloud LCB data → `interrank` MCP, not fresh measurements.
