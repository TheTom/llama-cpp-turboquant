#!/usr/bin/env python3
"""Track GPU VRAM and host RAM usage vs context length, with and without
block KV streaming, and plot both curves.

Unlike benchmark_kv_stream.py (which measures prefill/decode throughput and
therefore has to run a real prefill+decode pass per point), this script only
needs to load the model and construct the context - the KV/compute buffers
are sized at construction time, not grown lazily during inference - so each
point is just "start server, read memory, stop server," making it cheap
enough to sweep many more context points, and to run a same-context
non-streaming baseline alongside the streaming measurement for direct
comparison (including past the point where the baseline OOMs).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_kv_stream import (  # noqa: E402
    Server,
    clean_server_env,
    context_capacities,
    parse_token_count,
    query_gpu_memory,
    server_command,
    wait_for_release,
)

ROOT = Path(__file__).resolve().parents[1]


def read_process_rss_mib(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/status") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None
    return None


def read_system_ram_used_mib() -> float | None:
    fields: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    fields[key] = int(rest.strip().split()[0])
    except (FileNotFoundError, ValueError, IndexError):
        return None
    if "MemTotal" not in fields or "MemAvailable" not in fields:
        return None
    return (fields["MemTotal"] - fields["MemAvailable"]) / 1024.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-context", type=parse_token_count, required=True)
    parser.add_argument("--min-context", type=parse_token_count, default=8192)
    parser.add_argument("--context-step", type=parse_token_count, default=8192)
    parser.add_argument("--arena-mib", type=int, default=8192, help="shared arena size for the streaming leg")
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="skip the non-streaming (arena disabled) leg at each context",
    )
    parser.add_argument("--server", type=Path, default=ROOT / "build/bin/llama-server")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ubatch-size", type=int, default=256)
    parser.add_argument("--settle-seconds", type=float, default=2.0, help="wait after health-check before reading memory")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--port", type=int, default=12356)
    parser.add_argument("--startup-timeout", type=int, default=240)
    parser.add_argument("--release-timeout", type=int, default=90)
    parser.add_argument("--release-slack-mib", type=int, default=64)
    parser.add_argument(
        "--extra-server-arg", action="append", default=[], metavar="ARG",
        help="append a server argument (repeat; use --extra-server-arg=--flag)",
    )
    args = parser.parse_args(argv)
    args.model = args.model.resolve()
    args.server = args.server.resolve()
    args.trace_kv_stream = False  # Server/clean_server_env expect this attribute
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if args.min_context <= 0 or args.context_step <= 0:
        raise SystemExit("minimum context and context step must be positive")
    if args.min_context > args.max_context:
        raise SystemExit("minimum context must not exceed maximum context")
    if args.arena_mib <= 0:
        raise SystemExit("arena size must be positive")
    if args.ubatch_size > args.batch_size:
        raise SystemExit("ubatch size must not exceed batch size")


def measure_leg(
    args: argparse.Namespace,
    context_capacity: int,
    arena_mib: int,
    log_path: Path,
) -> dict:
    baseline_gpu = query_gpu_memory(args.nvidia_smi, args.gpu_index)
    try:
        server = Server(args, context_capacity, arena_mib, log_path)
    except RuntimeError as exc:
        return {"status": "failed", "error": str(exc)}
    try:
        time.sleep(args.settle_seconds)
        gpu = query_gpu_memory(args.nvidia_smi, args.gpu_index)
        rss_mib = read_process_rss_mib(server.process.pid)
        sysram_mib = read_system_ram_used_mib()
        return {
            "status": "ok",
            "vram_used_mib": gpu.used_mib,
            "vram_free_mib": gpu.free_mib,
            "process_rss_mib": rss_mib,
            "system_ram_used_mib": sysram_mib,
        }
    finally:
        server.stop()
        try:
            wait_for_release(args, baseline_gpu.used_mib)
        except RuntimeError:
            pass  # best-effort; the next point's own baseline read still catches reality


def write_csv(path: Path, rows: dict[int, dict]) -> None:
    fields = [
        "context_capacity",
        "arena_mib",
        "streaming_status",
        "streaming_vram_used_mib",
        "streaming_process_rss_mib",
        "streaming_system_ram_used_mib",
        "baseline_status",
        "baseline_vram_used_mib",
        "baseline_process_rss_mib",
        "baseline_system_ram_used_mib",
        "baseline_error",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for context in sorted(rows):
            writer.writerow({field: rows[context].get(field) for field in fields})


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Matplotlib is required. Install it with: python3 -m pip install matplotlib"
        ) from exc
    return plt


def first_failure_context(rows: dict[int, dict]) -> tuple[int, str] | None:
    for context in sorted(rows):
        row = rows[context]
        if row.get("baseline_status") == "failed":
            return context, row.get("baseline_error") or "failed"
    return None


def annotate_oom(ax, contexts: list[int], failure: tuple[int, str] | None) -> None:
    if failure is None:
        return
    context, error = failure
    x_oom = context / 1024
    ax.axvline(x_oom, color="#d62728", linestyle=":", linewidth=1.6, alpha=0.8)
    reason = "CUDA OOM" if "out of memory" in error.lower() else "crashed"
    ax.annotate(
        f"no-streaming {reason}\nat {context // 1024}Ki context",
        xy=(x_oom, 0.96), xycoords=("data", "axes fraction"),
        xytext=(6, -6), textcoords="offset points",
        color="#d62728", fontsize=9, va="top",
    )


def plot_results(output_dir: Path, rows: dict[int, dict], plt) -> None:
    if not rows:
        return
    contexts = sorted(rows)
    x = [context / 1024 for context in contexts]
    failure = first_failure_context(rows)

    def series(key: str) -> list[float | None]:
        return [rows[c].get(key) for c in contexts]

    def trim_trailing_none(xs: list[float], ys: list[float | None]) -> tuple[list[float], list[float]]:
        pairs = [(a, b) for a, b in zip(xs, ys) if b is not None]
        return [a for a, _ in pairs], [b for _, b in pairs]

    fig, (vram_ax, ram_ax) = plt.subplots(
        2, 1, figsize=(12.5, 9), sharex=True, constrained_layout=True,
    )

    vx, vy = trim_trailing_none(x, series("streaming_vram_used_mib"))
    vram_ax.plot(vx, vy, color="#1f77b4", marker="o", linewidth=2.4, label="Streaming (VRAM)")
    bx, by = trim_trailing_none(x, series("baseline_vram_used_mib"))
    if bx:
        vram_ax.plot(bx, by, color="#d62728", marker="x", linewidth=2.0, linestyle="--",
                     label="No streaming (VRAM)")
    annotate_oom(vram_ax, contexts, failure)
    vram_ax.set_title("KV cache memory footprint vs context length")
    vram_ax.set_ylabel("GPU VRAM used (MiB)")
    vram_ax.set_ylim(bottom=0)
    vram_ax.grid(True, alpha=0.25)
    vram_ax.legend(loc="best")

    rx, ry = trim_trailing_none(x, series("streaming_process_rss_mib"))
    ram_ax.plot(rx, ry, color="#2ca02c", marker="o", linewidth=2.4, label="Streaming (host RSS)")
    rbx, rby = trim_trailing_none(x, series("baseline_process_rss_mib"))
    if rbx:
        ram_ax.plot(rbx, rby, color="#9467bd", marker="x", linewidth=2.0, linestyle="--",
                     label="No streaming (host RSS)")
    annotate_oom(ram_ax, contexts, failure)
    ram_ax.set_xlabel("Configured context capacity (Ki tokens)")
    ram_ax.set_ylabel("Server process RSS (MiB)")
    ram_ax.set_ylim(bottom=0)
    ram_ax.grid(True, alpha=0.25)
    ram_ax.legend(loc="best")

    png_path = output_dir / "kv-memory-sweep.png"
    fig.savefig(png_path, dpi=180)
    fig.savefig(png_path.with_suffix(".svg"))
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    plt = require_matplotlib()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"

    capacities = context_capacities(args.max_context, args.min_context, args.context_step)
    print(f"Sweep: {len(capacities)} points, {capacities[0]} through {capacities[-1]} tokens, "
          f"arena={args.arena_mib} MiB, baseline={'off' if args.no_baseline else 'on'}", flush=True)

    rows: dict[int, dict] = {}
    baseline_alive = not args.no_baseline
    with jsonl_path.open("w") as jsonl:
        for i, context in enumerate(capacities, start=1):
            print(f"[{i}/{len(capacities)}] context capacity {context}", flush=True)
            row: dict = {"context_capacity": context, "arena_mib": args.arena_mib}

            streaming = measure_leg(
                args, context, args.arena_mib, logs_dir / f"streaming-{context}.log",
            )
            row["streaming_status"] = streaming["status"]
            if streaming["status"] == "ok":
                row["streaming_vram_used_mib"] = streaming["vram_used_mib"]
                row["streaming_process_rss_mib"] = streaming["process_rss_mib"]
                row["streaming_system_ram_used_mib"] = streaming["system_ram_used_mib"]
            else:
                print(f"  streaming leg failed: {streaming['error']}", flush=True)

            if baseline_alive:
                baseline = measure_leg(
                    args, context, 0, logs_dir / f"baseline-{context}.log",
                )
                row["baseline_status"] = baseline["status"]
                if baseline["status"] == "ok":
                    row["baseline_vram_used_mib"] = baseline["vram_used_mib"]
                    row["baseline_process_rss_mib"] = baseline["process_rss_mib"]
                    row["baseline_system_ram_used_mib"] = baseline["system_ram_used_mib"]
                else:
                    row["baseline_error"] = baseline["error"]
                    print(f"  baseline leg failed (expected past the VRAM ceiling): {baseline['error']}", flush=True)
                    baseline_alive = False  # monotonically larger context won't recover

            rows[context] = row
            jsonl.write(json.dumps(row) + "\n")
            jsonl.flush()
            write_csv(csv_path, rows)
            plot_results(output_dir, rows, plt)

    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")
    print(f"Plot:  {output_dir / 'kv-memory-sweep.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
