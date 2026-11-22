# interfer

Local MLX-LM inference server for Apple Silicon, with priority queuing,
thermal-aware scheduling and experiment hooks.

## Requirements

- Apple Silicon Mac (inference runs on Metal)
- Python 3.12+ and [uv](https://docs.astral.sh/uv/)

## Run

```bash
uv run python -m server --dry-run   # no model load; serves on port 8421
uv run python -m server             # real MLX inference
curl http://localhost:8421/health
```

A Starlette process handles HTTP and talks to a separate Metal subprocess over
a multiprocessing queue. Server code is in `server/`, benchmarks in
`benchmarks/`, tests in `tests/`.

See `CLAUDE.md` and `AGENTS.md` for development notes.
