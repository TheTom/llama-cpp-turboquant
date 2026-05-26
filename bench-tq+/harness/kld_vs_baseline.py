#!/usr/bin/env python3
"""
KL-divergence diagnostic — measures distributional drift introduced by
each TQ+ codec vs the fp16/fp16 baseline.

Workflow:
  1. Run llama-perplexity with f16/f16 + --save-all-logits → base.kld
  2. For each test codec, run llama-perplexity with --kl-divergence
     --kl-divergence-base base.kld
  3. Parse Mean KLD, Maximum KLD, percentiles (99.9, 99, 95, 90, ..., 1),
     same-top fraction, Δ p (mean difference in probability of the
     correct next token).

What this answers:
  - Mean KLD: average distributional drift per token
  - Max KLD: worst-case position (heavy-tail outlier?)
  - High percentiles (99%, 99.9%): is error concentrated in a few
    positions (suggests sink/recency would help) or spread uniformly?
  - Same-top: fraction of positions where argmax matches baseline.
    If this stays high (>99%), greedy decode is preserved even though
    sampled-temperature decode drifts.

Run:
  python kld_vs_baseline.py --model M.gguf --corpus C.raw \
      --codecs "q8_0/turbo3" "q8_0/turbo2" \
      --ctx 4096 --chunks 20 \
      --out results/kld.json
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


KLD_LINE_PATTERNS = [
    ("mean_kld",    re.compile(r"^Mean\s+KLD:\s+([\d.]+)")),
    ("median_kld",  re.compile(r"^Median\s+KLD:\s+([\d.]+)")),
    ("max_kld",     re.compile(r"^Maximum\s+KLD:\s+([\d.]+)")),
    ("kld_99_9",    re.compile(r"^99\.9%\s+KLD:\s+([\d.]+)")),
    ("kld_99_0",    re.compile(r"^99\.0%\s+KLD:\s+([\d.]+)")),
    ("kld_95_0",    re.compile(r"^95\.0%\s+KLD:\s+([\d.]+)")),
    ("kld_90_0",    re.compile(r"^90\.0%\s+KLD:\s+([\d.]+)")),
    ("kld_10_0",    re.compile(r"^10\.0%\s+KLD:\s+([\d.]+)")),
    ("kld_05_0",    re.compile(r"^\s*5\.0%\s+KLD:\s+([\d.]+)")),
    ("kld_01_0",    re.compile(r"^\s*1\.0%\s+KLD:\s+([\d.]+)")),
    ("same_top_pct",re.compile(r"Same top p:\s+([\d.]+)")),
    ("mean_pdiff",  re.compile(r"Mean Δp:\s+(-?[\d.]+)")),
    ("rms_pdiff",   re.compile(r"RMS Δp:\s+([\d.]+)")),
    ("max_pdiff",   re.compile(r"Maximum Δp:\s+([\d.]+)")),
    ("base_ppl",    re.compile(r"Base perplexity:\s+([\d.]+)")),
    ("ppl",         re.compile(r"Perplexity:\s+([\d.]+)")),
    ("ppl_ratio",   re.compile(r"^Perplexity ratio:\s+([\d.]+)")),
]


def parse_kld_output(text: str) -> dict:
    """Extract aggregate KLD metrics from llama-perplexity output."""
    metrics = {k: None for k, _ in KLD_LINE_PATTERNS}
    for line in text.splitlines():
        line = line.strip()
        for key, pat in KLD_LINE_PATTERNS:
            m = pat.search(line)
            if m and metrics[key] is None:
                try:
                    metrics[key] = float(m.group(1))
                except ValueError:
                    pass
    return metrics


@dataclass
class KldRun:
    label: str
    k_type: str
    v_type: str
    ctx: int
    chunks: int
    metrics: dict
    elapsed_s: float
    log_path: str
    ok: bool


def run_save_logits(perplexity_bin, model, corpus, ctx, chunks, threads,
                    out_logits, log_path, extra):
    """Run f16/f16 perplexity, save logits as binary kld base."""
    cmd = [
        str(perplexity_bin),
        "-m", str(model), "-f", str(corpus),
        "-c", str(ctx), "--chunks", str(chunks),
        "--cache-type-k", "f16", "--cache-type-v", "f16",
        "-ngl", "999", "-t", str(threads),
        "-fa", "on",
        "--kl-divergence-base", str(out_logits),
        *extra,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60 * 60)
    elapsed = time.time() - t0
    log_path.write_text(
        f"$ {' '.join(shlex.quote(c) for c in cmd)}\n# rc={proc.returncode}, elapsed={elapsed:.1f}s\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-3000:]}\n"
    )
    return proc.returncode == 0, elapsed


def run_kld_eval(perplexity_bin, model, corpus, k_type, v_type, ctx,
                 chunks, threads, base_logits, log_path, extra):
    """Run codec vs base logits, parse aggregate KLD metrics."""
    cmd = [
        str(perplexity_bin),
        "-m", str(model), "-f", str(corpus),
        "-c", str(ctx), "--chunks", str(chunks),
        "--cache-type-k", k_type, "--cache-type-v", v_type,
        "-ngl", "999", "-t", str(threads),
        "-fa", "on",
        "--kl-divergence",
        "--kl-divergence-base", str(base_logits),
        *extra,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60 * 60)
    elapsed = time.time() - t0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(
        f"$ {' '.join(shlex.quote(c) for c in cmd)}\n# rc={proc.returncode}, elapsed={elapsed:.1f}s\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr[-3000:]}\n"
    )
    metrics = parse_kld_output(out)
    return metrics, elapsed, proc.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--codecs", nargs="+", required=True,
                    help="codecs to KL-test vs f16/f16 baseline")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, default=REPO / "bench-tq+/logs")
    ap.add_argument("--base-logits", type=Path,
                    default=REPO / "bench-tq+/results/f16_base.kld",
                    help="path to save the f16 baseline logits (reused across codecs)")
    ap.add_argument("--perplexity-bin", type=Path, default=REPO / "build/bin/llama-perplexity")
    ap.add_argument("--extra-args", default="--fit off")
    ap.add_argument("--skip-base", action="store_true",
                    help="reuse existing base_logits file instead of regenerating")
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    extra = shlex.split(args.extra_args)

    # Step 1: generate or reuse base logits
    if not args.skip_base or not args.base_logits.exists():
        print(f"[step 1] generating f16/f16 baseline logits → {args.base_logits}", flush=True)
        log_path = args.log_dir / f"kld_base_c{args.ctx}_n{args.chunks}.log"
        ok, elapsed = run_save_logits(
            args.perplexity_bin, args.model, args.corpus,
            args.ctx, args.chunks, args.threads,
            args.base_logits, log_path, extra,
        )
        if not ok:
            print(f"  ✗ base logits run failed, see {log_path}")
            sys.exit(1)
        print(f"  ✓ base logits saved ({elapsed:.0f}s)")
    else:
        print(f"[step 1] reusing existing base logits at {args.base_logits}")

    # Step 2: KL-divergence vs base for each codec
    results = []
    for codec in args.codecs:
        k, v = codec.split("/")
        log_path = args.log_dir / f"kld_{k}-{v}_c{args.ctx}_n{args.chunks}.log"
        print(f"[step 2] {codec}", flush=True)
        metrics, elapsed, ok = run_kld_eval(
            args.perplexity_bin, args.model, args.corpus,
            k, v, args.ctx, args.chunks, args.threads,
            args.base_logits, log_path, extra,
        )
        results.append(asdict(KldRun(
            label=codec, k_type=k, v_type=v,
            ctx=args.ctx, chunks=args.chunks,
            metrics=metrics, elapsed_s=elapsed,
            log_path=str(log_path), ok=ok,
        )))
        args.out.write_text(json.dumps(results, indent=2))
        if ok:
            print(f"  ✓ mean_kld={metrics.get('mean_kld')} "
                  f"99%={metrics.get('kld_99_0')} "
                  f"max={metrics.get('max_kld')} "
                  f"same_top%={metrics.get('same_top_pct')}  "
                  f"elapsed={elapsed:.0f}s")
        else:
            print(f"  ✗ failed (see {log_path})")
        sys.stdout.flush()

    # Step 3: summary
    print("\nKL-divergence summary (lower = closer to fp16 baseline):")
    print(f"  {'codec':<18} {'mean':>10} {'med':>10} {'95%':>10} {'99%':>10} {'99.9%':>10} {'max':>10} {'top%':>8}")
    for r in results:
        m = r["metrics"]
        def fmt(v, w):
            if v is None: return "—".rjust(w)
            return f"{v:.4f}".rjust(w) if abs(v) < 100 else f"{v:.2f}".rjust(w)
        print(f"  {r['label']:<18} "
              f"{fmt(m.get('mean_kld'), 10)} "
              f"{fmt(m.get('median_kld'), 10)} "
              f"{fmt(m.get('kld_95_0'), 10)} "
              f"{fmt(m.get('kld_99_0'), 10)} "
              f"{fmt(m.get('kld_99_9'), 10)} "
              f"{fmt(m.get('max_kld'), 10)} "
              f"{fmt(m.get('same_top_pct'), 8)}")

    print("\n# DIAGNOSTIC INTERPRETATION:")
    print("# If max_kld and 99.9% kld are LARGE relative to median:")
    print("#   → error is concentrated in a small fraction of positions")
    print("#   → sink+recency selectively protects those positions, would help")
    print("# If max_kld ≈ 99% kld ≈ 95% kld:")
    print("#   → error is uniform across positions")
    print("#   → sink+recency wouldn't help; position-agnostic codec improvement needed")


if __name__ == "__main__":
    main()
