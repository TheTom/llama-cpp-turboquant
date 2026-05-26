#!/usr/bin/env python3
"""
Per-position KL divergence diagnostic — measures WHERE quantization
error concentrates along the sequence under TQ+ defaults.

This is the scientific gate before implementing sink/recency:
  - If KL is roughly flat across positions, sink+recency wouldn't help
  - If KL spikes at certain position classes (e.g. very low or very
    high positions), that's the empirical justification for selectively
    keeping those positions at fp16.

Methodology:
  1. For a fixed prompt of length L, generate (or extract) the next-token
     logits at every position p in [0, L) under codec X.
  2. Do the same under fp16/f16 baseline.
  3. Compute KL(p_baseline || p_codec) at each position.
  4. Bin / average / plot.

llama-perplexity computes per-token cross-entropy as it sweeps. We use
its `--logits-file` mode (if available) or our own implementation via
`llama-cli --logit-bias` / a custom invocation.

Simpler implementation: use llama-perplexity's `--all-logits` (if it
exists) or `--ppl-output`. Otherwise we leverage the per-token
cross-entropy that perplexity ALREADY computes — it's just per-chunk
averaged. We can extract per-token via the `--ppl-show-tokens` family
of flags or write a thin C wrapper.

Pragmatic approach: use perplexity's NLL-per-chunk output (one number
per chunk) and run multiple separate runs at DIFFERENT chunk sizes —
e.g., chunks of length 64, 256, 1K, 4K — to see how compression error
scales with context length. If chunks of length 4K have ~4× the
per-token NLL of chunks of length 64, recency window helps. If NLL
is flat across chunk sizes, recency window doesn't help.

This gives us the macro answer (does context-length affect quant
error?) without needing per-token logit dumps. The diagnostic is
weaker than full per-position KL but is implementable in one evening
using existing llama.cpp tools.

Usage:
    python per_position_kld.py --model M.gguf --corpus C.raw \
        --codecs "q8_0/turbo3" "f16/f16" \
        --chunk-sizes 256 1024 4096 16384 \
        --out results/kld_diagnostic.json
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


CHUNK_INLINE_RE = re.compile(r"\[(\d+)\](-?[\d.]+)")


@dataclass
class ChunkSizeResult:
    label: str
    k_type: str
    v_type: str
    chunk_size: int
    n_chunks: int
    final_ppl: float | None
    chunk_ppls: list[float]
    kv_total_mb: float | None
    elapsed_s: float
    log_path: str
    ok: bool


def run_ppl(
    perplexity_bin: Path,
    model: Path,
    corpus: Path,
    k_type: str,
    v_type: str,
    chunk_size: int,
    n_chunks: int,
    threads: int,
    log_path: Path,
    extra: list[str],
) -> ChunkSizeResult:
    cmd = [
        str(perplexity_bin),
        "-m", str(model),
        "-f", str(corpus),
        "-c", str(chunk_size),
        "--chunks", str(n_chunks),
        "--cache-type-k", k_type,
        "--cache-type-v", v_type,
        "-ngl", "999",
        "-t", str(threads),
        "-fa", "on",
        *extra,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60 * 60)
    elapsed = time.time() - t0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(
        f"$ {' '.join(shlex.quote(c) for c in cmd)}\n"
        f"# rc = {proc.returncode}\n# elapsed = {elapsed:.1f}s\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n",
    )
    final_ppl = None
    chunk_ppls: list[float] = []
    kv_mb = None
    for line in out.splitlines():
        if "Final estimate: PPL =" in line:
            try:
                final_ppl = float(line.split("PPL =")[1].split()[0])
            except (IndexError, ValueError):
                pass
        for m in CHUNK_INLINE_RE.finditer(line):
            chunk_ppls.append(float(m.group(2)))
        if "KV self size" in line:
            # `KV self size = X MiB, K (...): Y MiB, V (...): Z MiB`
            try:
                kv_mb = float(line.split("=")[1].split("MiB")[0].strip())
            except (IndexError, ValueError):
                pass
    return ChunkSizeResult(
        label=f"k={k_type}_v={v_type}_c{chunk_size}",
        k_type=k_type, v_type=v_type,
        chunk_size=chunk_size, n_chunks=n_chunks,
        final_ppl=final_ppl, chunk_ppls=chunk_ppls,
        kv_total_mb=kv_mb, elapsed_s=elapsed,
        log_path=str(log_path),
        ok=(proc.returncode == 0 and final_ppl is not None),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--codecs", nargs="+", required=True,
                    help="codec specs as 'k_type/v_type', e.g. 'f16/f16' 'q8_0/turbo3'")
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[256, 1024, 4096, 16384],
                    help="context lengths to sweep")
    ap.add_argument("--n-chunks", type=int, default=20,
                    help="number of chunks at each chunk-size")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, default=REPO / "bench-tq+/logs")
    ap.add_argument("--perplexity-bin", type=Path, default=REPO / "build/bin/llama-perplexity")
    ap.add_argument("--extra-args", default="--fit off")
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    extra = shlex.split(args.extra_args)

    results = []
    for codec_spec in args.codecs:
        k_type, v_type = codec_spec.split("/")
        for cs in args.chunk_sizes:
            log_path = args.log_dir / f"kld_{k_type}-{v_type}_c{cs}_n{args.n_chunks}.log"
            print(f"[run] {codec_spec}  chunk_size={cs}  n_chunks={args.n_chunks}")
            sys.stdout.flush()
            r = run_ppl(
                args.perplexity_bin, args.model, args.corpus,
                k_type, v_type, cs, args.n_chunks, args.threads,
                log_path, extra,
            )
            results.append(asdict(r))
            args.out.write_text(json.dumps(results, indent=2))
            if r.ok:
                print(f"  ✓ final_ppl={r.final_ppl:.4f}  per-chunk_n={len(r.chunk_ppls)}  "
                      f"kv={r.kv_total_mb}MB  elapsed={r.elapsed_s:.0f}s")
            else:
                print(f"  ✗ failed (see {log_path})")
            sys.stdout.flush()

    print(f"\nWrote {len(results)} rows to {args.out}")

    print("\nPPL vs chunk size matrix:")
    codec_specs = sorted(set((r["k_type"], r["v_type"]) for r in results))
    chunk_sizes = sorted(set(r["chunk_size"] for r in results))
    print(f"  {'codec':<20} " + " ".join(f"{cs:>10}" for cs in chunk_sizes))
    for k, v in codec_specs:
        row = []
        for cs in chunk_sizes:
            r = next((x for x in results if x["k_type"] == k and x["v_type"] == v and x["chunk_size"] == cs), None)
            row.append(f"{r['final_ppl']:.4f}" if r and r["final_ppl"] else "—")
        print(f"  {k+'/'+v:<20} " + " ".join(f"{c:>10}" for c in row))

    # If both baseline and a quantized codec are present, also print
    # ΔPPL vs baseline as a function of chunk size — the key diagnostic.
    baseline_key = ("f16", "f16")
    if baseline_key in codec_specs:
        print("\nΔPPL vs f16/f16 baseline (%):")
        print(f"  {'codec':<20} " + " ".join(f"{cs:>10}" for cs in chunk_sizes))
        for k, v in codec_specs:
            if (k, v) == baseline_key:
                continue
            row = []
            for cs in chunk_sizes:
                rb = next((x for x in results if x["k_type"] == "f16" and x["v_type"] == "f16" and x["chunk_size"] == cs and x["final_ppl"]), None)
                ro = next((x for x in results if x["k_type"] == k and x["v_type"] == v and x["chunk_size"] == cs and x["final_ppl"]), None)
                if rb and ro:
                    d = (ro["final_ppl"] - rb["final_ppl"]) / rb["final_ppl"] * 100
                    row.append(f"{d:+.2f}%")
                else:
                    row.append("—")
            print(f"  {k+'/'+v:<20} " + " ".join(f"{c:>10}" for c in row))

        print("\n# DIAGNOSTIC INTERPRETATION")
        print("# If ΔPPL grows with chunk_size, sink+recency would HELP")
        print("#   (because compression error accumulates with context length).")
        print("# If ΔPPL is flat or smaller at larger ctx, sink+recency wouldn't help much.")


if __name__ == "__main__":
    main()
