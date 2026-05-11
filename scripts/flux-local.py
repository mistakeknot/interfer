#!/usr/bin/env python3
"""flux-local — run flux-review lens fleet against a local interfer server.

Usage:
    flux-local DIFF_FILE [--lens correctness,safety,quality] [--server http://localhost:8421]

For each lens, POSTs to the server's /v1/chat/completions endpoint with the
lens's system prompt and the diff as user content. Fires all lenses in
parallel. Prints each lens's review separately.

Spike scope (Sylveste-k8c step 1):
- Plumbing only — does the bridge work, does the server batch?
- Quality calibration is a later step; for now any non-empty response is
  considered "the bridge works".
- 3 lenses default (correctness, safety, quality). Add more via --lens.

Doesn't depend on Claude Code or the Anthropic API. Runs entirely on the
local interfer server (localhost:8421).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path

import httpx

# Lens names map to filenames; restrict to safe chars to block path-traversal
# attacks like `--lens ../etc/passwd`. Found by the safety lens reviewing its
# own deployment infra (commit 9aed7ae, k8c spike 2026-05-10).
_LENS_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Look for the lens prompt cache automatically. Override with FLUX_LENS_DIR
# if you've moved the interflux plugin or want a custom directory.
DEFAULT_LENS_DIR = Path(
    os.environ.get(
        "FLUX_LENS_DIR",
        Path.home()
        / ".claude"
        / "plugins"
        / "cache"
        / "interagency-marketplace"
        / "interflux"
        / "0.2.69"
        / "agents"
        / "review",
    )
)


def load_lens_prompt(lens_name: str, lens_dir: Path) -> str:
    """Load `fd-<name>.md`, strip YAML frontmatter, return system prompt body.

    `lens_name` must match `[a-z0-9][a-z0-9_-]*` and the resolved path must
    stay inside `lens_dir`; both checks block path traversal via crafted
    lens names. Belt-and-suspenders: the regex is the primary defense and
    the resolved-path check catches symlinks pointing outside lens_dir.
    """
    if not _LENS_NAME_RE.match(lens_name):
        raise SystemExit(
            f"Invalid lens name: {lens_name!r} " f"(must match {_LENS_NAME_RE.pattern})"
        )
    path = lens_dir / f"fd-{lens_name}.md"
    resolved = path.resolve()
    lens_dir_resolved = lens_dir.resolve()
    try:
        resolved.relative_to(lens_dir_resolved)
    except ValueError:
        raise SystemExit(
            f"Lens path escapes lens_dir: {resolved} not under {lens_dir_resolved}"
        )
    if not path.exists():
        raise SystemExit(
            f"Lens prompt not found: {path}\n"
            f"  set FLUX_LENS_DIR to the correct interflux agents/review directory"
        )
    raw = path.read_text()
    # Strip YAML frontmatter — everything between the first --- and the second ---
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            return raw[end + 4 :].lstrip()
    return raw


async def run_lens(
    client: httpx.AsyncClient,
    server: str,
    model: str,
    lens_name: str,
    system_prompt: str,
    diff: str,
    max_tokens: int,
) -> dict:
    """Fire one lens. Returns timing + response text + finish_reason."""
    user_msg = (
        "Review the following diff. Apply your lens. List concrete findings; "
        "skip generic advice.\n\n"
        f"```diff\n{diff}\n```\n"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": True,  # interfer always streams; client-side SSE accumulation
    }
    t0 = time.monotonic()
    try:
        # interfer's /v1/chat/completions always returns text/event-stream.
        # Accumulate `delta.content` chunks; last chunk carries finish_reason
        # and (optionally) usage.
        import json as _json

        chunks: list[str] = []
        finish: str | None = None
        completion_tokens = 0
        async with client.stream(
            "POST",
            f"{server}/v1/chat/completions",
            json=payload,
            timeout=600.0,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = _json.loads(data)
                except _json.JSONDecodeError:
                    continue
                # Some OpenAI-compat servers emit errors mid-stream as
                # `{"error": {...}}` inside a 200-OK response. Without this
                # check we'd swallow the error and return empty content
                # (caught by the correctness lens, k8c spike 2026-05-10).
                if isinstance(obj.get("error"), dict):
                    err = obj["error"]
                    msg = err.get("message") or err.get("type") or str(err)
                    raise RuntimeError(f"server error mid-stream: {msg}")
                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if "content" in delta and delta["content"]:
                    chunks.append(delta["content"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    completion_tokens = usage["completion_tokens"]
        wall = time.monotonic() - t0
        content = "".join(chunks)
        return {
            "lens": lens_name,
            "wall_s": round(wall, 1),
            "finish_reason": finish,
            "completion_tokens": completion_tokens,
            "content": content,
            "error": None,
        }
    except httpx.HTTPStatusError as e:
        return {
            "lens": lens_name,
            "wall_s": round(time.monotonic() - t0, 1),
            "error": f"HTTP {e.response.status_code}: {e.response.text[:300]}",
            "content": "",
            "finish_reason": None,
            "completion_tokens": 0,
        }
    except Exception as e:
        return {
            "lens": lens_name,
            "wall_s": round(time.monotonic() - t0, 1),
            "error": f"{type(e).__name__}: {e}",
            "content": "",
            "finish_reason": None,
            "completion_tokens": 0,
        }


async def main_async(args: argparse.Namespace) -> int:
    diff_path = Path(args.diff_file)
    if not diff_path.exists():
        print(f"diff file not found: {diff_path}", file=sys.stderr)
        return 2
    diff = diff_path.read_text()
    lens_names = [s.strip() for s in args.lens.split(",") if s.strip()]
    lens_dir = Path(args.lens_dir)

    prompts = {name: load_lens_prompt(name, lens_dir) for name in lens_names}

    print(
        f"flux-local — {len(lens_names)} lenses, model={args.model}, "
        f"diff={diff_path.name} ({len(diff)} chars)",
        flush=True,
    )
    print(f"  lenses: {', '.join(lens_names)}", flush=True)
    print(f"  server: {args.server}", flush=True)
    print()

    t0 = time.monotonic()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *[
                run_lens(
                    client,
                    args.server,
                    args.model,
                    name,
                    prompts[name],
                    diff,
                    args.max_tokens,
                )
                for name in lens_names
            ]
        )
    total_wall = time.monotonic() - t0

    print(f"=== Summary (wall: {total_wall:.1f}s) ===")
    for r in results:
        if r["error"]:
            print(f"  [{r['lens']:15s}] ERROR ({r['wall_s']}s): {r['error']}")
        else:
            print(
                f"  [{r['lens']:15s}] {r['wall_s']:6.1f}s  "
                f"finish={r['finish_reason']:8s}  "
                f"tokens={r['completion_tokens']:5d}"
            )
    print()

    for r in results:
        sep = "=" * 72
        print(f"\n{sep}\n  Lens: {r['lens']}\n{sep}\n")
        if r["error"]:
            print(f"ERROR: {r['error']}")
        else:
            print(r["content"])

    return 0 if all(not r["error"] for r in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="flux-local")
    parser.add_argument("diff_file", help="Path to diff file (output of git diff)")
    parser.add_argument(
        "--lens",
        default="correctness,safety,quality",
        help="Comma-separated lens names (looks for fd-<name>.md)",
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8421",
        help="Interfer server base URL",
    )
    parser.add_argument(
        "--model",
        default="qwen3.6-35b-a3b-dwq",
        help="Model name to send in the request (must be loaded on the server)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="max_tokens per lens (default: 4096)",
    )
    parser.add_argument(
        "--lens-dir",
        default=str(DEFAULT_LENS_DIR),
        help=f"Directory containing fd-*.md prompts (default: {DEFAULT_LENS_DIR})",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
