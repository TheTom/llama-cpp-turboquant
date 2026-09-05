#!/usr/bin/env python3
"""Benchmark and plot a matrix of adaptive-streaming K/V cache types."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).with_name("benchmark_kv_stream.py")
DEFAULT_TYPES = ("bf16", "q8_0", "q4_0")


def parse_token_count(value: str) -> int:
    text = value.strip().lower()
    for suffix, multiplier in (
        ("kib", 1024),
        ("ki", 1024),
        ("k", 1024),
    ):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    else:
        multiplier = 1
    try:
        count = int(text) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid token count: {value}") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("token count must be positive")
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--server",
        type=Path,
        default=ROOT / "build-kv-cuda/bin/llama-server",
    )
    parser.add_argument("--min-context", type=parse_token_count, default=40 * 1024)
    parser.add_argument("--max-context", type=parse_token_count, default=160 * 1024)
    parser.add_argument("--context-step", type=parse_token_count, default=8 * 1024)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument(
        "--arena-mib", "--pool-mib",
        dest="arena_mib",
        type=int,
        help="fixed shared arena; omit to probe the maximum usable arena at every point",
    )
    parser.add_argument(
        "--types",
        default=",".join(DEFAULT_TYPES),
        help="comma-separated cache types used for both K and V",
    )
    parser.add_argument("--port", type=int, default=12355)
    parser.add_argument("--startup-timeout", type=int, default=240)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="result directory; reuse it to resume an interrupted matrix",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="skip measurements and regenerate outputs from completed pair CSVs",
    )
    return parser.parse_args(argv)


def cache_types(value: str) -> list[str]:
    result = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not result:
        raise SystemExit("at least one cache type is required")
    if len(set(result)) != len(result):
        raise SystemExit("cache types must not contain duplicates")
    return result


def pair_name(type_k: str, type_v: str) -> str:
    return f"k-{type_k}_v-{type_v}"


def runner_command(
    args: argparse.Namespace,
    type_k: str,
    type_v: str,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--model",
        str(args.model),
        "--server",
        str(args.server),
        "--min-context",
        str(args.min_context),
        "--max-context",
        str(args.max_context),
        "--context-step",
        str(args.context_step),
        "--decode-tokens",
        str(args.decode_tokens),
        "--cache-type-k",
        type_k,
        "--cache-type-v",
        type_v,
        "--trace-kv-stream",
        "--port",
        str(args.port),
        "--startup-timeout",
        str(args.startup_timeout),
        "--request-timeout",
        str(args.request_timeout),
        "--output-dir",
        str(output_dir),
    ]
    if args.arena_mib is not None:
        command.extend(("--fixed-arena-mib", str(args.arena_mib)))
    return command


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes")


def collect_rows(output_dir: Path, types: list[str]) -> list[dict]:
    rows = []
    for type_k in types:
        for type_v in types:
            path = output_dir / pair_name(type_k, type_v) / "results.csv"
            if not path.is_file():
                continue
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    row["cache_type_k"] = type_k
                    row["cache_type_v"] = type_v
                    row["pair"] = f"K={type_k.upper()}, V={type_v.upper()}"
                    if not row.get("arena_mib") and row.get("pool_mib"):
                        row["arena_mib"] = row["pool_mib"]
                    row["streaming_active"] = parse_bool(row.get("streaming_active"))
                    rows.append(row)
    return rows


def write_combined_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "cache_type_k",
        "cache_type_v",
        "pair",
        "context_capacity",
        "prompt_tokens",
        "decode_tokens",
        "arena_mib",
        "prefill_tps",
        "decode_tps",
        "prompt_ms",
        "predicted_ms",
        "wall_seconds",
        "vram_before_mib",
        "vram_after_mib",
        "vram_free_after_mib",
        "streaming_active",
        "stream_first_active_tokens",
        "stream_trace_samples",
        "stream_max_active_pages",
        "stream_min_resident_pages",
        "stream_max_ring_slots",
        "stream_repartitions",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                item["cache_type_k"],
                item["cache_type_v"],
                int(item["context_capacity"]),
            ),
        ):
            writer.writerow({field: row.get(field) for field in fields})


def write_onsets(path: Path, rows: list[dict], types: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "cache_type_k",
                "cache_type_v",
                "first_streaming_context",
                "first_streaming_context_kib",
                "first_stream_active_tokens",
            ),
        )
        writer.writeheader()
        for type_k in types:
            for type_v in types:
                pair_rows = sorted(
                    (
                        row
                        for row in rows
                        if row["cache_type_k"] == type_k
                        and row["cache_type_v"] == type_v
                        and row["streaming_active"]
                    ),
                    key=lambda row: int(row["context_capacity"]),
                )
                first = pair_rows[0] if pair_rows else None
                context = int(first["context_capacity"]) if first else None
                writer.writerow(
                    {
                        "cache_type_k": type_k,
                        "cache_type_v": type_v,
                        "first_streaming_context": context,
                        "first_streaming_context_kib": (
                            context / 1024 if context is not None else None
                        ),
                        "first_stream_active_tokens": (
                            first.get("stream_first_active_tokens") if first else None
                        ),
                    }
                )


def plot_results(path: Path, rows: list[dict], types: list[str]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pairs = [(type_k, type_v) for type_k in types for type_v in types]
    cmap = plt.get_cmap("tab10")
    fig, (prefill_ax, decode_ax) = plt.subplots(
        2,
        1,
        figsize=(15, 11),
        sharex=True,
        constrained_layout=True,
    )
    pair_handles = []
    for index, (type_k, type_v) in enumerate(pairs):
        pair_rows = sorted(
            (
                row for row in rows
                if row["cache_type_k"] == type_k and row["cache_type_v"] == type_v
            ),
            key=lambda row: int(row["context_capacity"]),
        )
        if not pair_rows:
            continue
        color = cmap(index % 10)
        label = f"K={type_k.upper()}, V={type_v.upper()}"
        pair_handles.append(Line2D([0], [0], color=color, lw=2.2, label=label))
        x = [int(row["context_capacity"]) / 1024 for row in pair_rows]
        for axis, metric, style in (
            (prefill_ax, "prefill_tps", "--"),
            (decode_ax, "decode_tps", "-"),
        ):
            y = [float(row[metric]) for row in pair_rows]
            axis.plot(x, y, color=color, linestyle=style, linewidth=2.0)
            inactive = [
                (x_value, y_value)
                for x_value, y_value, row in zip(x, y, pair_rows)
                if not row["streaming_active"]
            ]
            active = [
                (x_value, y_value)
                for x_value, y_value, row in zip(x, y, pair_rows)
                if row["streaming_active"]
            ]
            if inactive:
                axis.scatter(
                    *zip(*inactive), s=30, facecolors="white",
                    edgecolors=[color], linewidths=1.3, zorder=3,
                )
            if active:
                axis.scatter(
                    *zip(*active), s=34, facecolors=[color],
                    edgecolors="black", linewidths=0.55, zorder=4,
                )
                onset = active[0]
                axis.scatter(
                    [onset[0]], [onset[1]], marker="*", s=190,
                    facecolors=[color], edgecolors="black",
                    linewidths=0.8, zorder=5,
                )

    selected_types = " / ".join(cache_type.upper() for cache_type in types)
    prefill_ax.set_title(
        f"Adaptive KV streaming: {selected_types} cache-type matrix")
    prefill_ax.set_ylabel("Prefill speed (tokens/s)")
    decode_ax.set_ylabel("Decode speed (tokens/s)")
    decode_ax.set_xlabel("Configured context capacity (Ki tokens)")
    for axis in (prefill_ax, decode_ax):
        axis.grid(True, alpha=0.25)
        axis.set_ylim(bottom=0)

    state_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="white",
            markeredgecolor="black", label="Fully resident",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor="#777777",
            markeredgecolor="black", label="Streaming active",
        ),
        Line2D(
            [0], [0], marker="*", linestyle="none", markersize=13,
            markerfacecolor="#777777", markeredgecolor="black",
            label="First streaming point",
        ),
    ]
    prefill_ax.legend(handles=state_handles, loc="lower left")
    fig.legend(
        handles=pair_handles,
        loc="outside upper center",
        ncol=3,
        title="KV cache types",
    )
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if not args.server.is_file():
        raise SystemExit(f"server not found: {args.server}")
    if args.min_context > args.max_context:
        raise SystemExit("minimum context must not exceed maximum context")
    if min(args.context_step, args.decode_tokens) <= 0:
        raise SystemExit("step and decode tokens must be positive")
    if args.arena_mib is not None and args.arena_mib <= 0:
        raise SystemExit("fixed arena must be positive")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    types = cache_types(args.types)
    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = ROOT / "benchmarks/results" / f"kv-type-matrix-{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(type_k, type_v) for type_k in types for type_v in types]
    if not args.plot_only:
        for index, (type_k, type_v) in enumerate(pairs, start=1):
            pair_dir = args.output_dir / pair_name(type_k, type_v)
            print(
                f"\n=== [{index}/{len(pairs)}] K={type_k}, V={type_v} ===",
                flush=True,
            )
            subprocess.run(
                runner_command(args, type_k, type_v, pair_dir),
                cwd=ROOT,
                check=True,
            )
            rows = collect_rows(args.output_dir, types)
            write_combined_csv(args.output_dir / "kv-type-matrix.csv", rows)
            write_onsets(args.output_dir / "streaming-onsets.csv", rows, types)
            plot_results(args.output_dir / "kv-type-matrix.png", rows, types)

    rows = collect_rows(args.output_dir, types)
    if not rows:
        raise SystemExit("no completed measurements found")
    write_combined_csv(args.output_dir / "kv-type-matrix.csv", rows)
    write_onsets(args.output_dir / "streaming-onsets.csv", rows, types)
    plot_results(args.output_dir / "kv-type-matrix.png", rows, types)
    print(f"Results: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
