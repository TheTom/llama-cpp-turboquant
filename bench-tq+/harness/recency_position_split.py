#!/usr/bin/env python3
"""
Recency-vs-sink position split test — measures whether the per-token
quantization error concentrates at high attention-weight positions.

Without modifying llama.cpp, we can use a clever experimental design:
   for each chunk of size C, compute the per-token PPL averaged
   separately over the FIRST N_sink tokens, the LAST N_recency tokens,
   and the MIDDLE.

llama-perplexity emits per-chunk PPL but not per-token. So we run
perplexity at MULTIPLE chunk-size + N_skip configurations:

   Run 1: chunks of 4096 tokens, all averaged → baseline
   Run 2: chunks of 64 tokens, all averaged → only-recency (no context buildup)
   Run 3: chunks of 4 tokens, all averaged → first-N-only (no buildup)

If Run 3 PPL ≈ Run 1 PPL, sinks would NOT help.
If Run 2 PPL ≈ Run 1 PPL, recency would help A LOT (the only good positions are the recent ones).
If Run 2 PPL < Run 1 PPL substantially, recency dominates the error.

This is a coarse proxy for per-position KL — it answers the
qualitative direction question (where does compression error
concentrate?) without needing per-token logit dumps.
"""

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def run_ppl(perplexity_bin, model, corpus, k_type, v_type, ctx, n_chunks, threads, log_path, extra):
    cmd = [
        str(perplexity_bin),
        "-m", str(model), "-f", str(corpus),
        "-c", str(ctx), "--chunks", str(n_chunks),
        "--cache-type-k", k_type, "--cache-type-v", v_type,
        "-ngl", "999", "-t", str(threads),
        "-fa", "on", *extra,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60 * 60)
    elapsed = time.time() - t0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(
        f"$ {' '.join(shlex.quote(c) for c in cmd)}\n"
        f"# rc={proc.returncode} elapsed={elapsed:.1f}s\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-2000:]}\n",
    )
    final_ppl = None
    for line in out.splitlines():
        if "Final estimate: PPL =" in line:
            try:
                final_ppl = float(line.split("PPL =")[1].split()[0])
            except (IndexError, ValueError):
                pass
    return final_ppl, elapsed, proc.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--codecs", nargs="+", required=True)
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[256, 1024, 4096])
    ap.add_argument("--n-chunks", type=int, default=20)
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
    for codec in args.codecs:
        k, v = codec.split("/")
        for cs in args.chunk_sizes:
            log_path = args.log_dir / f"rps_{k}-{v}_c{cs}.log"
            print(f"[run] {codec} ctx={cs} n_chunks={args.n_chunks}", flush=True)
            ppl, elapsed, ok = run_ppl(
                args.perplexity_bin, args.model, args.corpus,
                k, v, cs, args.n_chunks, args.threads, log_path, extra,
            )
            results.append({
                "codec": codec, "k_type": k, "v_type": v,
                "ctx": cs, "n_chunks": args.n_chunks,
                "final_ppl": ppl, "elapsed_s": elapsed, "ok": ok,
                "log_path": str(log_path),
            })
            args.out.write_text(json.dumps(results, indent=2))
            print(f"  → PPL={ppl}  elapsed={elapsed:.0f}s", flush=True)

    # Summary table
    codecs = sorted(set(r["codec"] for r in results))
    ctxs = sorted(set(r["ctx"] for r in results))
    print(f"\n  {'codec':<18} " + " ".join(f"{c:>10}" for c in ctxs))
    for codec in codecs:
        row = []
        for cs in ctxs:
            r = next((x for x in results if x["codec"] == codec and x["ctx"] == cs and x["final_ppl"]), None)
            row.append(f"{r['final_ppl']:.4f}" if r else "—")
        print(f"  {codec:<18} " + " ".join(f"{c:>10}" for c in row))

    base = next((r for r in results if r["codec"] == "f16/f16"), None)
    if base:
        print(f"\nΔPPL% vs f16/f16 (positive = quant codec worse):")
        print(f"  {'codec':<18} " + " ".join(f"{c:>10}" for c in ctxs))
        for codec in codecs:
            if codec == "f16/f16":
                continue
            row = []
            for cs in ctxs:
                rb = next((x for x in results if x["codec"] == "f16/f16" and x["ctx"] == cs and x["final_ppl"]), None)
                ro = next((x for x in results if x["codec"] == codec and x["ctx"] == cs and x["final_ppl"]), None)
                if rb and ro:
                    d = (ro["final_ppl"] - rb["final_ppl"]) / rb["final_ppl"] * 100
                    row.append(f"{d:+.2f}%")
                else:
                    row.append("—")
            print(f"  {codec:<18} " + " ".join(f"{c:>10}" for c in row))

        print("\n# Interpretation:")
        print("# If ΔPPL is small at ctx=256 and large at ctx=4096:")
        print("#   → quant error accumulates with context. Recency window WOULD help.")
        print("# If ΔPPL is similar across ctx sizes:")
        print("#   → quant error is per-token, not accumulating. Sink+recency unlikely to help.")


if __name__ == "__main__":
    main()
