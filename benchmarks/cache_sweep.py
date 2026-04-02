#!/usr/bin/env python3
"""Flash-MoE cache sweep: Q3 vs 4-bit across malloc-cache and cache-io-split configs.

Invokes the flash-moe binary directly (not via HTTP server) for precise per-token
timing. Requires shaders.metal at ./metal_infer/shaders.metal relative to the
flash-moe repo root — the script auto-discovers and cds there.

Usage:
  python benchmarks/cache_sweep.py --dry-run          # print commands only
  python benchmarks/cache_sweep.py                     # full 12-config sweep
  python benchmarks/cache_sweep.py --quant q3          # Q3 only (6 configs)
  python benchmarks/cache_sweep.py --malloc-cache 0    # single cache size

Environment:
  FLASHMOE_BINARY  — path to infer binary (default: ~/projects/flash-moe/metal_infer/infer)
  FLASHMOE_MODEL   — model directory (default: ~/Models/flash_mlx_4bit)

Outputs TSV with columns:
  quant, malloc_cache, cache_io_split, startup_s, ttft_ms,
  mean_tps, median_tps, min_tps, max_tps, hit_rate_pct,
  cold_misses, eviction_misses, expert_io_pct, status
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


PROMPT = "Explain quantum computing in one concise paragraph."
GEN_TOKENS = 64

MALLOC_CACHE_VALUES = [0, 2500, 5000]
CACHE_IO_SPLIT_VALUES = [0, 4]
QUANT_MODES = ["4bit", "q3"]


def find_flashmoe_root(binary: str) -> Path:
    """Discover the flash-moe repo root from the binary path.

    The binary must be run from the repo root because it expects
    shaders.metal at ./metal_infer/shaders.metal.
    """
    binary_path = Path(binary).resolve()
    # binary is typically at <repo>/metal_infer/infer
    candidate = binary_path.parent.parent
    shader = candidate / "metal_infer" / "shaders.metal"
    if shader.is_file():
        return candidate
    # fallback: check binary's dir directly
    if (binary_path.parent / "shaders.metal").is_file():
        return binary_path.parent
    raise SystemExit(
        f"Cannot find shaders.metal relative to binary {binary}.\n"
        f"Checked: {candidate}/metal_infer/shaders.metal"
    )


@dataclass
class BenchResult:
    quant: str = ""
    malloc_cache: int = 0
    cache_io_split: int = 0
    startup_s: float = 0.0
    ttft_ms: float = 0.0
    mean_tps: float = 0.0
    median_tps: float = 0.0
    min_tps: float = 0.0
    max_tps: float = 0.0
    hit_rate_pct: float = 0.0
    cold_misses: int = 0
    eviction_misses: int = 0
    expert_io_pct: float = 0.0
    status: str = "ok"
    per_token_tps: list[float] = field(default_factory=list)
    raw_output: str = ""


def build_cmd(
    binary: str,
    model: str,
    quant: str,
    malloc_cache: int,
    cache_io_split: int,
    gen_tokens: int,
    prompt: str,
    gguf_args: list[str],
) -> list[str]:
    cmd = [
        binary,
        "--model",
        model,
        "--prompt",
        prompt,
        "--tokens",
        str(gen_tokens),
        "--malloc-cache",
        str(malloc_cache),
        "--timing",
        "--cache-telemetry",
    ]
    if quant == "q3":
        cmd.append("--q3-experts")
    if cache_io_split > 0:
        cmd.extend(["--cache-io-split", str(cache_io_split)])
    cmd.extend(gguf_args)
    return cmd


def parse_output(text: str) -> BenchResult:
    r = BenchResult(raw_output=text)

    # Per-token throughput: "token_id=... (796 ms, 1.26 tok/s)"
    tps_matches = re.findall(r"\((\d+)\s*ms,\s*([0-9.]+)\s*tok/s\)", text)
    if tps_matches:
        r.per_token_tps = [float(m[1]) for m in tps_matches]
        r.mean_tps = statistics.mean(r.per_token_tps)
        r.median_tps = statistics.median(r.per_token_tps)
        r.min_tps = min(r.per_token_tps)
        r.max_tps = max(r.per_token_tps)

    # TTFT
    ttft = re.search(r"TTFT:\s*(\d+)\s*ms", text)
    if ttft:
        r.ttft_ms = float(ttft.group(1))

    # Cache hit rate
    hit = re.search(r"(\d+)\s*hits,\s*(\d+)\s*misses\s*\(([0-9.]+)%\s*hit rate\)", text)
    if hit:
        r.hit_rate_pct = float(hit.group(3))

    # Cache telemetry: "Miss breakdown: cold 6404 (82.0%), eviction 1410 (18.0%)"
    miss_line = re.search(
        r"cold\s+(\d+)\s*\([^)]*\),\s*eviction\s+(\d+)", text, re.IGNORECASE
    )
    if miss_line:
        r.cold_misses = int(miss_line.group(1))
        r.eviction_misses = int(miss_line.group(2))

    # Expert I/O percentage
    io_pct = re.search(r"Expert I/O.*?([0-9.]+)%", text)
    if io_pct:
        r.expert_io_pct = float(io_pct.group(1))

    # Total time
    total = re.search(r"Total time:\s*([0-9.]+)\s*s", text)
    if total:
        r.startup_s = float(total.group(1))

    return r


TSV_COLUMNS = [
    "quant",
    "malloc_cache",
    "cache_io_split",
    "startup_s",
    "ttft_ms",
    "mean_tps",
    "median_tps",
    "min_tps",
    "max_tps",
    "hit_rate_pct",
    "cold_misses",
    "eviction_misses",
    "expert_io_pct",
    "status",
]


def main() -> int:
    default_binary = os.environ.get(
        "FLASHMOE_BINARY",
        os.path.expanduser("~/projects/flash-moe/metal_infer/infer"),
    )
    default_model = os.environ.get(
        "FLASHMOE_MODEL",
        os.path.expanduser("~/Models/flash_mlx_4bit"),
    )

    parser = argparse.ArgumentParser(
        description="Q3 vs 4-bit cache sweep benchmark (direct binary invocation)",
    )
    parser.add_argument(
        "--binary", default=default_binary, help="Path to flash-moe infer binary"
    )
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--gen-tokens", type=int, default=GEN_TOKENS)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument(
        "--timeout", type=int, default=600, help="Per-run timeout in seconds"
    )
    parser.add_argument(
        "--malloc-cache",
        nargs="+",
        type=int,
        default=MALLOC_CACHE_VALUES,
        help="malloc-cache values to sweep",
    )
    parser.add_argument(
        "--cache-io-split",
        nargs="+",
        type=int,
        default=CACHE_IO_SPLIT_VALUES,
        help="cache-io-split values to sweep",
    )
    parser.add_argument(
        "--quant",
        nargs="+",
        default=QUANT_MODES,
        choices=QUANT_MODES,
        help="Quantization modes to benchmark",
    )
    parser.add_argument(
        "--gguf-embedding",
        default=os.path.expanduser("~/Models/flash_mlx_4bit/gguf/embedding_q8_0.bin"),
    )
    parser.add_argument(
        "--gguf-lm-head",
        default=os.path.expanduser("~/Models/flash_mlx_4bit/gguf/lm_head_q6.bin"),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running"
    )
    args = parser.parse_args()

    # Resolve paths
    binary = str(Path(args.binary).resolve())
    if not Path(binary).is_file():
        raise SystemExit(f"Binary not found: {binary}")
    flashmoe_root = find_flashmoe_root(binary)

    bench_dir = Path(__file__).resolve().parent
    out_path = bench_dir / f"results_{time.strftime('%Y%m%d_%H%M%S')}.tsv"
    logs_dir = bench_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Build common GGUF overlay args
    gguf_args: list[str] = []
    if args.gguf_embedding and Path(args.gguf_embedding).is_file():
        gguf_args.extend(["--gguf-embedding", args.gguf_embedding])
    if args.gguf_lm_head and Path(args.gguf_lm_head).is_file():
        gguf_args.extend(["--gguf-lm-head", args.gguf_lm_head])

    configs = [
        (quant, mc, cis)
        for quant in args.quant
        for mc in args.malloc_cache
        for cis in args.cache_io_split
    ]

    print(f"Sweep: {len(configs)} configurations")
    print(f"Binary: {binary}")
    print(f"CWD: {flashmoe_root}")
    print(f"Model: {args.model}")
    print(f"Output: {out_path}")
    print()

    results: list[BenchResult] = []

    for i, (quant, mc, cis) in enumerate(configs):
        label = f"{quant}_mc{mc}_cis{cis}"
        print(f"[{i+1}/{len(configs)}] {label} ...", end=" ", flush=True)

        cmd = build_cmd(
            binary,
            args.model,
            quant,
            mc,
            cis,
            args.gen_tokens,
            args.prompt,
            gguf_args,
        )

        if args.dry_run:
            print(f"\n  $ {' '.join(cmd)}")
            continue

        log_file = logs_dir / f"{label}.log"
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(flashmoe_root),
                text=True,
                capture_output=True,
                timeout=args.timeout,
                check=False,
            )
            elapsed = time.time() - t0
            output = proc.stdout + "\n" + proc.stderr
            log_file.write_text(f"$ {' '.join(cmd)}\n\n{output}")

            if proc.returncode != 0:
                r = BenchResult(
                    quant=quant,
                    malloc_cache=mc,
                    cache_io_split=cis,
                    status=f"exit:{proc.returncode}",
                    raw_output=output,
                )
                print(f"FAIL (exit {proc.returncode}, {elapsed:.1f}s)")
            else:
                r = parse_output(output)
                r.quant = quant
                r.malloc_cache = mc
                r.cache_io_split = cis
                print(
                    f"{r.mean_tps:.2f} tok/s, {r.hit_rate_pct:.1f}% hits ({elapsed:.1f}s)"
                )

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            r = BenchResult(
                quant=quant,
                malloc_cache=mc,
                cache_io_split=cis,
                status="timeout",
                raw_output="",
            )
            print(f"TIMEOUT ({elapsed:.1f}s)")

        results.append(r)

    if args.dry_run:
        return 0

    # Write TSV
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "quant": r.quant,
                    "malloc_cache": r.malloc_cache,
                    "cache_io_split": r.cache_io_split,
                    "startup_s": f"{r.startup_s:.1f}",
                    "ttft_ms": f"{r.ttft_ms:.0f}",
                    "mean_tps": f"{r.mean_tps:.3f}",
                    "median_tps": f"{r.median_tps:.3f}",
                    "min_tps": f"{r.min_tps:.3f}",
                    "max_tps": f"{r.max_tps:.3f}",
                    "hit_rate_pct": f"{r.hit_rate_pct:.1f}",
                    "cold_misses": r.cold_misses,
                    "eviction_misses": r.eviction_misses,
                    "expert_io_pct": f"{r.expert_io_pct:.1f}",
                    "status": r.status,
                }
            )

    print(f"\nResults written to {out_path}")

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Config':<25} {'mean_tps':>10} {'hit_rate':>10} {'expert_io':>10}")
    print("-" * 60)
    for r in results:
        label = f"{r.quant}_mc{r.malloc_cache}_cis{r.cache_io_split}"
        if r.status == "ok":
            print(
                f"{label:<25} {r.mean_tps:>10.2f} {r.hit_rate_pct:>9.1f}% {r.expert_io_pct:>9.1f}%"
            )
        else:
            print(f"{label:<25} {'':>10} {'':>10} {r.status:>10}")

    ok_results = [r for r in results if r.status == "ok" and r.mean_tps > 0]
    if ok_results:
        best = max(ok_results, key=lambda r: r.mean_tps)
        print(
            f"\nBest: {best.quant}_mc{best.malloc_cache}_cis{best.cache_io_split} "
            f"at {best.mean_tps:.2f} tok/s ({best.hit_rate_pct:.1f}% hits)"
        )
        if best.mean_tps < 5.0:
            print(f"\n⚠  Best warm throughput ({best.mean_tps:.2f}) < 5 tok/s.")
            print("   Re-run best config with --freq for expert frequency profiling.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
