# Local-inference lens overrides

Slimmed lens prompts tuned for local 35B-class models (Qwen3.6-35B-A3B).
The canonical lenses in `interagency-marketplace/interflux/agents/review/`
are sized for Opus and assume full Claude Code tool access — they tend
to degenerate on smaller models when:

- The prompt contains multiple language sections and the model can't
  filter to the one relevant to the diff
- The prompt asks the model to "read CLAUDE.md, AGENTS.md, ..." but the
  model has no filesystem access
- The prompt is abstract enough that the model loses task focus and
  drifts into self-instruction loops (see quality lens degeneration in
  the 2026-05-10 spike result)

## Usage

```sh
uv run python scripts/flux-local.py <diff> \
  --lens-dir scripts/lens-local \
  --lens correctness,safety,quality \
  --model /Users/sma/Models/Qwen3.6-35B-A3B-4bit
```

If a lens isn't in this directory, fall back to the default plugin
cache (use `FLUX_LENS_DIR` to override the default location).

## Lenses here

| File | Differences from canonical |
|---|---|
| `fd-quality.md` | Python-only; dropped Go/TS/Shell/Rust sections; explicit "one-line-per-finding" output schema; explicit "no preambles" instruction |

## Add a new override

Copy the canonical lens from
`~/.claude/plugins/cache/interagency-marketplace/interflux/0.2.69/agents/review/fd-<name>.md`,
keep the YAML frontmatter, then:

1. Drop sections irrelevant to your diff's language
2. Specify a strict output schema (one-line-per-finding works well)
3. Add explicit "do not restate your role" / "no preambles" instructions
4. End with "Stop after listing findings"

Test by re-running the spike against the same diff and eyeballing
whether the output is now usable.
