#!/usr/bin/env python3
"""
TQ+ KV-cache codec A/B perplexity sweep.

Runs `llama-perplexity` against the same model + corpus across multiple
cache-type-k / cache-type-v configurations and captures:
  - Final perplexity
  - KV cache memory (parsed from llama startup log)
  - Wall-clock elapsed time
  - First-chunk PPL (proxy for prefill quality)

Designed to test the SHARD-inspired hypothesis: keeping sink tokens
(positions [0, N_SINK)) and the recency window (last N_REC positions)
at fp16 — rather than turbo-quantized — should close most of the
PPL gap vs full fp16 baseline at low memory cost. Tested against
TQ+'s actual default (q8_0/turbo3 + auto-Boundary V + Sparse V),
not against a strawman.

Usage:
    python run_ppl_sweep.py --model /path/to/model.gguf \
                            --corpus /path/to/wiki.test.raw \
                            --chunks 80 \
                            --ctx 4096 \
                            --out results/baseline.json

Outputs JSON to --out with one row per (k_type, v_type) combination.
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
DEFAULT_PERPLEXITY = REPO / "build/bin/llama-perplexity"
DEFAULT_BENCH = REPO / "build/bin/llama-bench"


@dataclass
class RunResult:
    label: str
    k_type: str
    v_type: str
    ctx: int
    chunks: int
    threads: int
    extra_args: str
    elapsed_s: float
    final_ppl: float | None
    first_chunk_ppl: float | None
    kv_total_mb: float | None
    kv_k_mb: float | None
    kv_v_mb: float | None
    eval_tps: float | None
    prompt_tps: float | None
    log_path: str
    ok: bool
    note: str


PPL_FINAL_RE = re.compile(r"Final estimate:\s+PPL\s*=\s*([0-9.]+)")
PPL_CHUNK_RE = re.compile(r"^\s*1\s+\d+\s+([0-9.]+)")
KV_SIZE_RE = re.compile(
    r"KV self size\s*=\s*([0-9.]+)\s*MiB,\s*K\s*\(\S+\):\s*([0-9.]+)\s*MiB,\s*V\s*\(\S+\):\s*([0-9.]+)\s*MiB"
)
EVAL_TPS_RE = re.compile(r"eval time =\s*[\d.]+\s*ms /\s*\d+\s*runs\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
PROMPT_TPS_RE = re.compile(r"prompt eval time =\s*[\d.]+\s*ms /\s*\d+\s*tokens\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)")


def parse_ppl_output(stdout: str, stderr: str) -> dict:
    """Extract metrics from llama-perplexity output. Either stream may carry the info."""
    combined = stdout + "\n" + stderr
    out = {
        "final_ppl": None,
        "first_chunk_ppl": None,
        "kv_total_mb": None,
        "kv_k_mb": None,
        "kv_v_mb": None,
        "eval_tps": None,
        "prompt_tps": None,
    }
    if m := PPL_FINAL_RE.search(combined):
        out["final_ppl"] = float(m.group(1))
    for line in combined.splitlines():
        if m := PPL_CHUNK_RE.match(line):
            out["first_chunk_ppl"] = float(m.group(1))
            break
    if m := KV_SIZE_RE.search(combined):
        out["kv_total_mb"] = float(m.group(1))
        out["kv_k_mb"] = float(m.group(2))
        out["kv_v_mb"] = float(m.group(3))
    if m := EVAL_TPS_RE.search(combined):
        out["eval_tps"] = float(m.group(1))
    if m := PROMPT_TPS_RE.search(combined):
        out["prompt_tps"] = float(m.group(1))
    return out


def run_one(
    perplexity_bin: Path,
    model: Path,
    corpus: Path,
    k_type: str,
    v_type: str,
    ctx: int,
    chunks: int,
    threads: int,
    log_dir: Path,
    extra_args: list[str],
    label: str | None = None,
) -> RunResult:
    label = label or f"k={k_type}_v={v_type}"
    log_path = log_dir / f"ppl_{label}_c{ctx}_n{chunks}.log"

    cmd = [
        str(perplexity_bin),
        "-m", str(model),
        "-f", str(corpus),
        "-c", str(ctx),
        "--chunks", str(chunks),
        "--cache-type-k", k_type,
        "--cache-type-v", v_type,
        "-ngl", "999",
        "-t", str(threads),
        "-fa", "on",  # flash-attention; required by some turbo paths
        *extra_args,
    ]
    print(f"[run] {label}  ({k_type}/{v_type})  ctx={ctx} chunks={chunks}")
    print(f"      {' '.join(shlex.quote(c) for c in cmd)}")

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60 * 60,  # 1h max per run
            check=False,
        )
        elapsed = time.time() - t0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        log_path.write_text(
            f"$ {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"# rc = {proc.returncode}\n"
            f"# elapsed = {elapsed:.1f}s\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
        )
        m = parse_ppl_output(stdout, stderr)
        return RunResult(
            label=label,
            k_type=k_type,
            v_type=v_type,
            ctx=ctx,
            chunks=chunks,
            threads=threads,
            extra_args=" ".join(extra_args),
            elapsed_s=elapsed,
            final_ppl=m["final_ppl"],
            first_chunk_ppl=m["first_chunk_ppl"],
            kv_total_mb=m["kv_total_mb"],
            kv_k_mb=m["kv_k_mb"],
            kv_v_mb=m["kv_v_mb"],
            eval_tps=m["eval_tps"],
            prompt_tps=m["prompt_tps"],
            log_path=str(log_path),
            ok=(proc.returncode == 0 and m["final_ppl"] is not None),
            note="" if proc.returncode == 0 else f"rc={proc.returncode}",
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        log_path.write_text(f"# TIMEOUT after {elapsed:.0f}s\n")
        return RunResult(
            label=label, k_type=k_type, v_type=v_type, ctx=ctx,
            chunks=chunks, threads=threads, extra_args=" ".join(extra_args),
            elapsed_s=elapsed, final_ppl=None, first_chunk_ppl=None,
            kv_total_mb=None, kv_k_mb=None, kv_v_mb=None,
            eval_tps=None, prompt_tps=None,
            log_path=str(log_path), ok=False, note="timeout"
        )


# ── Default codec matrix ────────────────────────────────────────────
# Ordered conservative → aggressive; each row is a (label, k, v).
# Mirrors the ladder in TQ+ README plus a few stress points.
DEFAULT_MATRIX = [
    ("f16-f16",   "f16",   "f16"),
    ("f16-turbo4","f16",   "turbo4"),
    ("q8-turbo4", "q8_0",  "turbo4"),
    ("q8-turbo3", "q8_0",  "turbo3"),    # TQ+ recommended default
    ("q8-turbo2", "q8_0",  "turbo2"),    # auto-Boundary V engages
    ("turbo3-turbo3", "turbo3", "turbo3"),  # discouraged but useful as upper-bound aggression
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--ctx", type=int, default=4096,
                    help="context length (also chunk size for perplexity).")
    ap.add_argument("--chunks", type=int, default=80,
                    help="number of perplexity chunks (more = lower variance).")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, default=REPO / "bench-tq+/logs")
    ap.add_argument("--perplexity-bin", type=Path, default=DEFAULT_PERPLEXITY)
    ap.add_argument("--matrix", choices=["default", "smoke", "all"], default="default")
    ap.add_argument("--extra-args", default="",
                    help="extra args appended to llama-perplexity (single quoted string).")
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.matrix == "smoke":
        matrix = [DEFAULT_MATRIX[0], DEFAULT_MATRIX[3]]  # f16/f16 + q8/turbo3
    else:
        matrix = DEFAULT_MATRIX
    extra_list = shlex.split(args.extra_args) if args.extra_args else []

    results = []
    for label, k_type, v_type in matrix:
        r = run_one(
            perplexity_bin=args.perplexity_bin,
            model=args.model,
            corpus=args.corpus,
            k_type=k_type,
            v_type=v_type,
            ctx=args.ctx,
            chunks=args.chunks,
            threads=args.threads,
            log_dir=args.log_dir,
            extra_args=extra_list,
            label=label,
        )
        results.append(asdict(r))
        # Save incrementally so partial progress survives a crash.
        args.out.write_text(json.dumps(results, indent=2))
        if r.ok:
            print(f"  ✓ ppl={r.final_ppl:.4f}  kv={r.kv_total_mb}MB  "
                  f"prompt_tps={r.prompt_tps}  eval_tps={r.eval_tps}  "
                  f"elapsed={r.elapsed_s:.0f}s")
        else:
            print(f"  ✗ FAILED  ({r.note})  see {r.log_path}")
        sys.stdout.flush()

    print(f"\nWrote {len(results)} rows to {args.out}")

    print("\nSummary (vs f16/f16 baseline):")
    print(f"  {'label':<18} {'ppl':>10} {'Δppl':>8} {'kv_MB':>8} {'p_tps':>8} {'e_tps':>8}")
    base_ppl = next((r["final_ppl"] for r in results if r["k_type"] == "f16" and r["v_type"] == "f16" and r["final_ppl"]), None)
    for r in results:
        ppl = r["final_ppl"]
        d = ((ppl - base_ppl) / base_ppl * 100) if (ppl and base_ppl) else None
        ppl_s = f"{ppl:.4f}" if ppl else "—"
        d_s = f"{d:+.2f}%" if d is not None else "—"
        kv_s = f"{r['kv_total_mb']:.0f}" if r["kv_total_mb"] else "—"
        pt = f"{r['prompt_tps']:.0f}" if r["prompt_tps"] else "—"
        et = f"{r['eval_tps']:.1f}" if r["eval_tps"] else "—"
        print(f"  {r['label']:<18} {ppl_s:>10} {d_s:>8} {kv_s:>8} {pt:>8} {et:>8}")


if __name__ == "__main__":
    main()
