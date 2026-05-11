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
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

import httpx

# Lens names map to filenames; restrict to safe chars to block path-traversal
# attacks like `--lens ../etc/passwd`. Found by the safety lens reviewing its
# own deployment infra (commit 9aed7ae, k8c spike 2026-05-10).
_LENS_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Patterns to redact from stdout when printing review content. The local model
# may quote pieces of the diff back at us, and if the diff contains a secret
# the model will faithfully reproduce it. Defense-in-depth flagged by the
# safety lens on the 2026-05-10 spike.
_SECRET_RE = re.compile(
    r"\b(sk-or-v1-[A-Za-z0-9_-]{30,}|sk-ant-[A-Za-z0-9_-]{30,}|sk-proj-[A-Za-z0-9_-]{30,}|ghp_[A-Za-z0-9_-]{30,}|github_pat_[A-Za-z0-9_-]{30,}|AKIA[0-9A-Z]{16})"
)

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _redact_secrets(text: str) -> str:
    """Mask known API-key / token patterns before printing review content."""
    return _SECRET_RE.sub(
        lambda m: f"<REDACTED:{m.group(1)[:7]}...{len(m.group(1))} chars>", text
    )


class LensResult(TypedDict):
    """Structured return value from `run_lens`.

    Typed alias so callers don't have to guess at the dict shape and so
    static checkers can flag missed fields. Promoted from a bare `dict`
    return hint after the quality-lens slim run flagged it (k8c spike,
    commit 5dad6cf).
    """

    lens: str
    wall_s: float
    finish_reason: str | None
    completion_tokens: int
    content: str
    error: str | None


class FluxLocalError(Exception):
    """Domain exception for flux-local helper failures.

    Helpers raise this instead of SystemExit so callers can catch them
    via standard `except` handling. The CLI entry point in main() is the
    only place that converts these into stderr messages and exit codes.

    The MAJOR finding from the quality-lens slim run on 2026-05-11 flagged
    SystemExit-from-a-helper as bypassing standard exception handling; this
    class is the fix.
    """


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


def load_lens_prompt(lens_name: str, lens_dirs: list[Path]) -> str:
    """Load `fd-<name>.md` from the first dir in `lens_dirs` that has it.

    Strips YAML frontmatter, returns the system prompt body. Each dir is
    validated: `lens_name` must match `[a-z0-9][a-z0-9_-]*` and the resolved
    path must stay inside the dir (belt-and-suspenders against symlink
    escapes). First match wins, so put overrides (e.g., 35B-tuned lenses
    in scripts/lens-local/) ahead of the canonical plugin cache.
    """
    if not _LENS_NAME_RE.match(lens_name):
        raise FluxLocalError(
            f"Invalid lens name: {lens_name!r} (must match {_LENS_NAME_RE.pattern})"
        )
    tried: list[Path] = []
    for lens_dir in lens_dirs:
        path = lens_dir / f"fd-{lens_name}.md"
        resolved = path.resolve()
        lens_dir_resolved = lens_dir.resolve()
        try:
            resolved.relative_to(lens_dir_resolved)
        except ValueError:
            raise FluxLocalError(
                f"Lens path escapes lens_dir: {resolved} not under {lens_dir_resolved}"
            )
        tried.append(path)
        if path.exists():
            raw = path.read_text()
            if raw.startswith("---"):
                end = raw.find("\n---", 3)
                if end != -1:
                    return raw[end + 4 :].lstrip()
            return raw
    raise FluxLocalError(
        f"Lens prompt not found: fd-{lens_name}.md\n"
        f"  searched: {', '.join(str(p) for p in tried)}\n"
        f"  set FLUX_LENS_DIR to override the canonical directory"
    )


async def run_lens(
    client: httpx.AsyncClient,
    server: str,
    model: str,
    lens_name: str,
    system_prompt: str,
    diff: str,
    max_tokens: int,
) -> LensResult:
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
        # Quality lens at temperature 0.3 was getting stuck in content loops
        # on small models — it would write the same 8-line analysis multiple
        # times without committing to the SEVERITY:file:line schema. Lowering
        # to 0.1 reduces drift; combined with example findings in the slim
        # prompt (scripts/lens-local/fd-quality.md) it now commits to output.
        # Sylveste-k8c step I, 2026-05-11.
        "temperature": 0.1,
        "stream": True,  # interfer always streams; client-side SSE accumulation
    }
    t0 = time.monotonic()
    try:
        # interfer's /v1/chat/completions always returns text/event-stream.
        # Accumulate `delta.content` chunks; last chunk carries finish_reason
        # and (optionally) usage.
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
                    obj = json.loads(data)
                except json.JSONDecodeError:
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
    # Build the lens-dir fallback chain: overrides first, then canonical.
    # Lets us ship local-tuned variants in scripts/lens-local/ while still
    # falling back to the full plugin cache for any lens not yet slimmed.
    lens_dirs: list[Path] = []
    if args.override_lens_dir:
        lens_dirs.append(Path(args.override_lens_dir))
    lens_dirs.append(Path(args.lens_dir))

    prompts = {name: load_lens_prompt(name, lens_dirs) for name in lens_names}

    # Warn (don't block) when sending the diff to a non-loopback server.
    # Local-inference flux-review is meant to keep the diff on the laptop;
    # an external server URL might be intentional (e.g., a trusted internal
    # interfer host), but flag it so it's not accidental. Flagged by the
    # safety lens on the 2026-05-10 spike.
    from urllib.parse import urlparse

    parsed_server = urlparse(args.server)
    if parsed_server.hostname and parsed_server.hostname not in _LOOPBACK_HOSTS:
        print(
            f"WARNING: --server {args.server!r} is not loopback "
            f"({sorted(_LOOPBACK_HOSTS)}); diff content will leave this machine.",
            file=sys.stderr,
            flush=True,
        )

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
            # The model may echo diff content back; redact known token
            # patterns before printing so secrets in the diff don't leak
            # to stdout/log files.
            print(_redact_secrets(r["content"]))

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
        help=f"Directory containing canonical fd-*.md prompts (default: {DEFAULT_LENS_DIR})",
    )
    parser.add_argument(
        "--override-lens-dir",
        default=None,
        help=(
            "Optional first-priority dir for slimmed/tuned lens variants. "
            "Lenses found here win over --lens-dir. Useful for local-model overrides."
        ),
    )
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except FluxLocalError as e:
        # Library-level errors translate to a clean stderr + exit 2 here.
        # Helpers raise this rather than SystemExit so they remain testable.
        print(f"flux-local: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
