#!/usr/bin/env python3
"""Run and plot an automatic context-matched adaptive KV streaming sweep."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CONTEXT_STEP = 8192
UVM_ENV_NAMES = (
    "GGML_CUDA_ENABLE_UNIFIED_MEMORY",
    "GGML_CUDA_PREFER_MODEL_WEIGHTS",
    "GGML_CUDA_PREFER_KV_HOST",
    "GGML_CUDA_KV_ACCESSED_BY_GPU",
)

KV_STREAM_TRACE_RE = re.compile(
    r"kv_stream_adapt: active (\d+), resident (\d+), ring (\d+), "
    r"samples (\d+), misses (\d+), copy busy ([0-9.]+)%, peak (\d+)"
)



def parse_token_count(value: str) -> int:
    text = value.strip().lower()
    multiplier = 1
    for suffix, factor in (
        ("kib", 1024),
        ("ki", 1024),
        ("k", 1024),
        ("mib", 1024 * 1024),
        ("mi", 1024 * 1024),
        ("m", 1024 * 1024),
    ):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            multiplier = factor
            break
    try:
        result = int(text) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid token count: {value}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("token count must be positive")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True, help="Qwen3.8-27B GGUF model")
    parser.add_argument(
        "--max-context",
        type=parse_token_count,
        required=True,
        help="largest configured context capacity, for example 192K or 196608",
    )
    parser.add_argument(
        "--min-context",
        type=parse_token_count,
        default=CONTEXT_STEP,
        help="smallest configured context capacity",
    )
    parser.add_argument(
        "--context-step",
        type=parse_token_count,
        default=CONTEXT_STEP,
        help="context-capacity increment",
    )
    parser.add_argument(
        "--server",
        type=Path,
        default=ROOT / "build/bin/llama-server",
        help="adaptive KV streaming llama-server binary",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="result directory; reuse it to resume an interrupted sweep",
    )
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="logical maximum prompt batch size",
    )
    parser.add_argument(
        "--ubatch-size", type=int, default=256,
        help="physical maximum batch size; must not exceed --batch-size",
    )
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q4_0")
    parser.add_argument(
        "--probe-arena-mib", "--probe-pool-mib",
        dest="probe_arena_mib", type=int, default=1536,
        help="initial total-arena size used to measure remaining VRAM",
    )
    parser.add_argument(
        "--probe-arena-step-mib", type=int, default=256,
        help="increase applied when the initial arena cannot fit prefill",
    )
    parser.add_argument(
        "--arena-step-mib", "--pool-step-mib",
        dest="arena_step_mib", type=int, default=32,
    )
    parser.add_argument(
        "--arena-backoff-mib", "--pool-backoff-mib",
        dest="arena_backoff_mib", type=int, default=64,
    )
    parser.add_argument(
        "--arena-retries", "--pool-retries",
        dest="arena_retries", type=int, default=8,
    )
    parser.add_argument(
        "--max-arena-mib", "--max-pool-mib",
        dest="max_arena_mib", type=int,
    )
    parser.add_argument(
        "--fixed-arena-mib", "--fixed-pool-mib",
        dest="fixed_arena_mib",
        type=int,
        help="use exactly this shared arena size and skip per-context probing",
    )
    parser.add_argument(
        "--trace-kv-stream",
        action="store_true",
        help="enable and parse adaptive KV residency trace logging",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--cuda-visible-devices",
        help="CUDA_VISIBLE_DEVICES value for the server; defaults to the inherited value",
    )
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--port", type=int, default=12355)
    parser.add_argument("--startup-timeout", type=int, default=240)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--release-timeout", type=int, default=90)
    parser.add_argument("--release-slack-mib", type=int, default=64)
    parser.add_argument("--prompt-suffix", default="The capital of France is")
    parser.add_argument("--fill-token-id", type=int)
    parser.add_argument(
        "--extra-server-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="append a server argument (repeat; use --extra-server-arg=--flag)",
    )
    args = parser.parse_args(argv)
    args.model = args.model.resolve()
    args.server = args.server.resolve()
    return args


def context_capacities(
    max_context: int,
    min_context: int = CONTEXT_STEP,
    step: int = CONTEXT_STEP,
) -> list[int]:
    capacities = list(range(min_context, max_context + 1, step))
    if not capacities or capacities[-1] != max_context:
        capacities.append(max_context)
    return capacities


@dataclass(frozen=True)
class GpuMemory:
    total_mib: int
    used_mib: int
    free_mib: int


def query_gpu_memory(nvidia_smi: str, gpu_index: int) -> GpuMemory:
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "-i",
                str(gpu_index),
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"cannot run nvidia-smi executable: {nvidia_smi}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    try:
        total, used, free = (
            int(part.strip()) for part in result.stdout.splitlines()[0].split(",")
        )
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout!r}") from exc
    return GpuMemory(total, used, free)


def estimate_arena_mib(
    probe_arena_mib: int,
    free_mib: int,
    step_mib: int,
    max_arena_mib: int | None = None,
) -> int:
    candidate = probe_arena_mib + free_mib
    candidate = candidate // step_mib * step_mib
    if max_arena_mib is not None:
        candidate = min(candidate, max_arena_mib // step_mib * step_mib)
    if candidate < probe_arena_mib:
        raise RuntimeError(f"only {free_mib} MiB is free after the probe")
    return candidate


def http_json(url: str, payload: dict | None, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc


def clean_server_env(
    cuda_visible_devices: str | None,
    trace_kv_stream: bool = False,
) -> dict[str, str]:
    env = os.environ.copy()
    for name in UVM_ENV_NAMES:
        env.pop(name, None)
    env.pop("GGML_CUDA_KV_STREAM_FIXED_RING_SLOTS", None)
    env.pop("LLAMA_KV_STREAM_TRACE", None)
    if trace_kv_stream:
        env["LLAMA_KV_STREAM_TRACE"] = "1"
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def server_command(
    args: argparse.Namespace,
    context_capacity: int,
    arena_mib: int,
) -> list[str]:
    return [
        str(args.server),
        "-m",
        str(args.model),
        "--alias",
        "adaptive-kv-stream",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--ctx-size",
        str(context_capacity),
        "-fa",
        "on",
        "-ctk",
        args.cache_type_k,
        "-ctv",
        args.cache_type_v,
        "-ngl",
        "all",
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "-np",
        "1",
        "--no-mmproj",
        "--no-warmup",
        "--fit",
        "off",
        "--reasoning-format",
        "none",
        "--kv-stream-arena-mib",
        str(arena_mib),
        *args.extra_server_arg,
    ]


class Server:
    def __init__(
        self,
        args: argparse.Namespace,
        context_capacity: int,
        arena_mib: int,
        log_path: Path,
    ) -> None:
        self.port = args.port
        self.log_path = log_path
        self.log_file = log_path.open("wb")
        self.process: subprocess.Popen | None = None
        try:
            self.process = subprocess.Popen(
                server_command(args, context_capacity, arena_mib),
                cwd=args.server.parent,
                env=clean_server_env(
                    args.cuda_visible_devices,
                    args.trace_kv_stream,
                ),
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            self.log_file.close()
            raise

        deadline = time.monotonic() + args.startup_timeout
        last_error = "server did not become ready"
        while time.monotonic() < deadline:
            status = self.process.poll()
            if status is not None:
                self.log_file.flush()
                log_tail = self.log_tail()
                self.stop()
                raise RuntimeError(f"server exited with status {status}: {log_tail}")
            try:
                health = http_json(self.url("/health"), None, 2)
                if health.get("status") == "ok":
                    return
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(f"{last_error}: {self.log_tail()}")

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def log_tail(self, lines: int = 30) -> str:
        try:
            return "\n".join(
                self.log_path.read_text(errors="replace").splitlines()[-lines:]
            )
        except OSError:
            return ""

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if not self.log_file.closed:
            self.log_file.close()


def wait_for_release(
    args: argparse.Namespace,
    baseline_used_mib: int,
) -> None:
    deadline = time.monotonic() + args.release_timeout
    while time.monotonic() < deadline:
        memory = query_gpu_memory(args.nvidia_smi, args.gpu_index)
        if memory.used_mib <= baseline_used_mib + args.release_slack_mib:
            return
        time.sleep(1)
    memory = query_gpu_memory(args.nvidia_smi, args.gpu_index)
    raise RuntimeError(
        f"GPU memory did not return to baseline: "
        f"{memory.used_mib} MiB used, baseline {baseline_used_mib} MiB"
    )


def prepare_server(
    args: argparse.Namespace,
    server: Server,
) -> tuple[list[int], int]:
    tokenized = http_json(
        server.url("/tokenize"),
        {"content": args.prompt_suffix, "add_special": False},
        30,
    )
    suffix = tokenized.get("tokens")
    if not suffix or not all(isinstance(token, int) for token in suffix):
        raise RuntimeError(f"unexpected tokenize response: {tokenized}")
    fill_token_id = (
        args.fill_token_id if args.fill_token_id is not None else suffix[0]
    )
    http_json(
        server.url("/completion"),
        {
            "prompt": [fill_token_id] * args.ubatch_size,
            "n_predict": 4,
            "ignore_eos": True,
            "cache_prompt": False,
            "temperature": 0,
            "reasoning_format": "none",
            "response_fields": ["timings"],
        },
        args.request_timeout,
    )
    return suffix, fill_token_id


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def load_measurements(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.is_file():
        return rows
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("type") == "measurement" and row.get("status") == "ok":
                rows[int(row["context_capacity"])] = row
    return rows


def load_metadata(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("type") == "metadata":
                return row
    return None


def resume_signature(args: argparse.Namespace, capacities: list[int]) -> dict:
    return {
        "model": str(args.model.resolve()),
        "server": str(args.server.resolve()),
        "max_context": args.max_context,
        "min_context": args.min_context,
        "context_step": args.context_step,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "fixed_arena_mib": args.fixed_arena_mib,
        "trace_kv_stream": args.trace_kv_stream,
        "capacities": capacities,
        "decode_tokens": args.decode_tokens,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "probe_arena_mib": args.probe_arena_mib,
        "probe_arena_step_mib": args.probe_arena_step_mib,
        "arena_step_mib": args.arena_step_mib,
        "arena_backoff_mib": args.arena_backoff_mib,
        "arena_retries": args.arena_retries,
        "max_arena_mib": args.max_arena_mib,
        "gpu_index": args.gpu_index,
        "cuda_visible_devices": args.cuda_visible_devices,
        "fill_token_id": args.fill_token_id,
        "prompt_suffix": args.prompt_suffix,
        "extra_server_args": args.extra_server_arg,
    }


def validate_resume(metadata: dict | None, signature: dict, path: Path) -> None:
    if metadata is None:
        raise SystemExit(f"existing result file has no metadata record: {path}")
    mismatches = [
        key for key, value in signature.items() if metadata.get(key) != value
    ]
    if mismatches:
        details = ", ".join(
            f"{key}={metadata.get(key)!r} (requested {signature[key]!r})"
            for key in mismatches
        )
        raise SystemExit(
            f"cannot resume {path} with different settings: {details}"
        )


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def probe_arena(
    args: argparse.Namespace,
    context_capacity: int,
    baseline_used_mib: int,
    logs_dir: Path,
    results_path: Path,
) -> int:
    arena_mib = args.probe_arena_mib
    last_error = "arena probe did not run"
    for attempt in range(args.arena_retries + 1):
        print(
            f"[{context_capacity}] VRAM probe attempt {attempt + 1}: "
            f"arena={arena_mib} MiB",
            flush=True,
        )
        server: Server | None = None
        try:
            server = Server(
                args,
                context_capacity,
                arena_mib,
                logs_dir / (
                    f"context-{context_capacity}-probe-arena-{arena_mib}.log"
                ),
            )
            prepare_server(args, server)
            memory = query_gpu_memory(args.nvidia_smi, args.gpu_index)
            selected_arena = estimate_arena_mib(
                arena_mib,
                memory.free_mib,
                args.arena_step_mib,
                args.max_arena_mib,
            )
            append_jsonl(
                results_path,
                {
                    "type": "arena_probe",
                    "context_capacity": context_capacity,
                    "probe_arena_mib": arena_mib,
                    "selected_arena_mib": selected_arena,
                    "vram_total_mib": memory.total_mib,
                    "vram_used_mib": memory.used_mib,
                    "vram_free_mib": memory.free_mib,
                },
            )
            return selected_arena
        except Exception as exc:
            last_error = str(exc)
            append_jsonl(
                results_path,
                {
                    "type": "arena_probe_attempt",
                    "status": "failed",
                    "context_capacity": context_capacity,
                    "arena_mib": arena_mib,
                    "attempt": attempt + 1,
                    "error": last_error[-4000:],
                },
            )
        finally:
            if server is not None:
                server.stop()
            wait_for_release(args, baseline_used_mib)

        next_arena = arena_mib + args.probe_arena_step_mib
        if args.max_arena_mib is not None and next_arena > args.max_arena_mib:
            break
        arena_mib = next_arena

    raise RuntimeError(
        f"context {context_capacity} could not start an arena probe: {last_error}"
    )


def parse_kv_stream_trace(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    samples = []
    for match in KV_STREAM_TRACE_RE.finditer(text):
        active_tokens = int(match.group(1))
        resident_pages = int(match.group(2))
        samples.append(
            {
                "active_tokens": active_tokens,
                "active_pages": (active_tokens + 255) // 256,
                "resident_pages": resident_pages,
                "ring_slots": int(match.group(3)),
            }
        )
    streamed = [
        sample
        for sample in samples
        if sample["active_pages"] > sample["resident_pages"]
    ]
    return {
        "streaming_active": bool(streamed),
        "stream_first_active_tokens": (
            min(sample["active_tokens"] for sample in streamed)
            if streamed
            else None
        ),
        "stream_trace_samples": len(samples),
        "stream_max_active_pages": max(
            (sample["active_pages"] for sample in samples), default=None
        ),
        "stream_min_resident_pages": min(
            (sample["resident_pages"] for sample in samples), default=None
        ),
        "stream_max_ring_slots": max(
            (sample["ring_slots"] for sample in samples), default=None
        ),
        "stream_repartitions": text.count("adaptive KV partition:"),
    }




def run_measurement(
    args: argparse.Namespace,
    context_capacity: int,
    arena_mib: int,
    baseline_used_mib: int,
    logs_dir: Path,
) -> dict:
    prompt_tokens = context_capacity - args.decode_tokens
    server: Server | None = None
    try:
        server = Server(
            args,
            context_capacity,
            arena_mib,
            logs_dir / f"context-{context_capacity}-arena-{arena_mib}.log",
        )
        suffix, fill_token_id = prepare_server(args, server)
        prefix_count = prompt_tokens - len(suffix)
        if prefix_count < 0:
            raise RuntimeError("context capacity is too small for the prompt")
        prompt = [fill_token_id] * prefix_count + suffix
        before = query_gpu_memory(args.nvidia_smi, args.gpu_index)
        started = time.monotonic()
        response = http_json(
            server.url("/completion"),
            {
                "prompt": prompt,
                "n_predict": args.decode_tokens,
                "ignore_eos": True,
                "cache_prompt": False,
                "temperature": 0,
                "seed": 1,
                "reasoning_format": "none",
                "response_fields": ["timings"],
            },
            args.request_timeout,
        )
        timings = response.get("timings") or {}
        if timings.get("predicted_n") != args.decode_tokens:
            raise RuntimeError(
                f"incomplete decode: expected {args.decode_tokens}, "
                f"received {timings.get('predicted_n')}"
            )
        after = query_gpu_memory(args.nvidia_smi, args.gpu_index)
        measurement = {
            "type": "measurement",
            "status": "ok",
            "context_capacity": context_capacity,
            "prompt_tokens": prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "arena_mib": arena_mib,
            "fill_token_id": fill_token_id,
            "prompt_ms": timings.get("prompt_ms"),
            "prefill_tps": timings.get("prompt_per_second"),
            "predicted_ms": timings.get("predicted_ms"),
            "decode_tps": timings.get("predicted_per_second"),
            "wall_seconds": time.monotonic() - started,
            "vram_before_mib": before.used_mib,
            "vram_after_mib": after.used_mib,
            "vram_free_after_mib": after.free_mib,
        }
        if args.trace_kv_stream:
            measurement.update(parse_kv_stream_trace(server.log_path))
        return measurement
    finally:
        if server is not None:
            server.stop()
        wait_for_release(args, baseline_used_mib)


def benchmark_with_backoff(
    args: argparse.Namespace,
    context_capacity: int,
    selected_arena_mib: int,
    baseline_used_mib: int,
    logs_dir: Path,
    results_path: Path,
) -> dict:
    arena_mib = selected_arena_mib
    last_error = "benchmark did not run"
    for attempt in range(args.arena_retries + 1):
        print(
            f"[{context_capacity}] benchmark attempt {attempt + 1}: "
            f"arena={arena_mib} MiB",
            flush=True,
        )
        try:
            return run_measurement(
                args,
                context_capacity,
                arena_mib,
                baseline_used_mib,
                logs_dir,
            )
        except Exception as exc:
            last_error = str(exc)
            append_jsonl(
                results_path,
                {
                    "type": "measurement_attempt",
                    "status": "failed",
                    "context_capacity": context_capacity,
                    "arena_mib": arena_mib,
                    "attempt": attempt + 1,
                    "error": last_error[-4000:],
                },
            )
            next_arena = (
                (arena_mib - args.arena_backoff_mib)
                // args.arena_step_mib
                * args.arena_step_mib
            )
            if next_arena < args.probe_arena_mib:
                break
            arena_mib = next_arena
    raise RuntimeError(
        f"context {context_capacity} failed after arena backoff: {last_error}"
    )


def write_csv(path: Path, rows: dict[int, dict]) -> None:
    fields = [
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
        for context in sorted(rows):
            writer.writerow({field: rows[context].get(field) for field in fields})


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Matplotlib is required. Install it with: "
            "python3 -m pip install matplotlib"
        ) from exc
    return plt


def plot_results(output_dir: Path, rows: dict[int, dict], plt) -> None:
    if not rows:
        return
    contexts = sorted(rows)
    x = [context / 1024 for context in contexts]

    fig, (decode_ax, arena_ax) = plt.subplots(
        2,
        1,
        figsize=(12.5, 9),
        height_ratios=(3, 1),
        sharex=True,
        constrained_layout=True,
    )
    prefill_ax = decode_ax.twinx()
    decode_ax.plot(
        x,
        [rows[context]["decode_tps"] for context in contexts],
        color="#1f77b4",
        marker="o",
        linewidth=2.4,
        label="Decode speed",
    )
    prefill_ax.plot(
        x,
        [rows[context]["prefill_tps"] for context in contexts],
        color="#ff7f0e",
        marker="o",
        linestyle="--",
        linewidth=2.2,
        label="Prefill speed",
    )
    arena_ax.plot(
        x,
        [rows[context]["arena_mib"] for context in contexts],
        color="#666666",
        marker="o",
        linewidth=1.8,
        label="Selected shared arena",
    )

    decode_ax.set_title("Adaptive KV streaming context sweep")
    decode_ax.set_ylabel("Decode speed (tokens/s)")
    prefill_ax.set_ylabel("Prefill speed (tokens/s)")
    arena_ax.set_xlabel("Configured context capacity (Ki tokens)")
    arena_ax.set_ylabel("Arena (MiB)")
    decode_ax.set_ylim(bottom=0)
    prefill_ax.set_ylim(bottom=0)
    arena_ax.set_ylim(bottom=0)
    decode_ax.grid(True, alpha=0.25)
    arena_ax.grid(True, alpha=0.25)
    handles_a, labels_a = decode_ax.get_legend_handles_labels()
    handles_b, labels_b = prefill_ax.get_legend_handles_labels()
    decode_ax.legend(handles_a + handles_b, labels_a + labels_b, loc="best")
    arena_ax.legend(loc="best")

    png_path = output_dir / "kv-stream-sweep.png"
    fig.savefig(png_path, dpi=180)
    fig.savefig(png_path.with_suffix(".svg"))
    plt.close(fig)


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if not args.server.is_file() or not os.access(args.server, os.X_OK):
        raise SystemExit(f"server is not executable: {args.server}")
    numeric_positive = (
        args.decode_tokens,
        args.batch_size,
        args.ubatch_size,
        args.probe_arena_mib,
        args.probe_arena_step_mib,
        args.arena_step_mib,
        args.arena_backoff_mib,
        args.startup_timeout,
        args.request_timeout,
        args.release_timeout,
    )
    if any(value <= 0 for value in numeric_positive):
        raise SystemExit("decode, batch, arena, and timeout settings must be positive")
    if args.min_context <= 0 or args.context_step <= 0:
        raise SystemExit("minimum context and context step must be positive")
    if args.min_context > args.max_context:
        raise SystemExit("minimum context must not exceed maximum context")
    if args.ubatch_size > args.batch_size:
        raise SystemExit("ubatch size must not exceed batch size")
    if args.fixed_arena_mib is not None and args.fixed_arena_mib <= 0:
        raise SystemExit("fixed arena must be positive")
    if (
        args.arena_retries < 0
        or args.release_slack_mib < 0
    ):
        raise SystemExit("retry count and release slack must not be negative")
    if args.max_arena_mib is not None and args.max_arena_mib < args.probe_arena_mib:
        raise SystemExit("maximum arena must be at least the probe arena")
    if args.gpu_index < 0 or not 1 <= args.port <= 65535:
        raise SystemExit("GPU index or port is invalid")
    if args.max_context < CONTEXT_STEP:
        raise SystemExit("maximum context must be at least 8192")
    if args.decode_tokens >= CONTEXT_STEP:
        raise SystemExit("decode token count must be smaller than 8192")
    if not args.prompt_suffix:
        raise SystemExit("prompt suffix must not be empty")
    if args.fill_token_id is not None and args.fill_token_id < 0:
        raise SystemExit("fill token ID must not be negative")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    plt = require_matplotlib()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir is None:
        args.output_dir = (
            ROOT / "benchmarks/results" / f"adaptive-kv-sweep-{stamp}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    csv_path = args.output_dir / "results.csv"

    baseline = query_gpu_memory(args.nvidia_smi, args.gpu_index)
    capacities = context_capacities(
        args.max_context,
        args.min_context,
        args.context_step,
    )
    signature = resume_signature(args, capacities)
    if results_path.exists():
        validate_resume(load_metadata(results_path), signature, results_path)
    rows = load_measurements(results_path)

    if not results_path.exists():
        append_jsonl(
            results_path,
            {
                "type": "metadata",
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "revision": git_revision(),
                **signature,
                "cache_type_k": args.cache_type_k,
                "cache_type_v": args.cache_type_v,
                "flash_attention": True,
                "parallel": 1,
                "baseline_vram_used_mib": baseline.used_mib,
                "baseline_vram_total_mib": baseline.total_mib,
                "uvm": False,
            },
        )

    print(
        f"GPU baseline: {baseline.used_mib}/{baseline.total_mib} MiB used",
        flush=True,
    )
    print(
        f"Sweep: {len(capacities)} points, {args.min_context} through "
        f"{args.max_context} tokens, K={args.cache_type_k}, V={args.cache_type_v}",
        flush=True,
    )

    interrupted = False
    failed = False
    try:
        for index, context_capacity in enumerate(capacities, start=1):
            if context_capacity in rows:
                print(
                    f"[{index}/{len(capacities)}] {context_capacity}: "
                    "already complete",
                    flush=True,
                )
                continue
            print(
                f"[{index}/{len(capacities)}] context capacity "
                f"{context_capacity}",
                flush=True,
            )
            try:
                if args.fixed_arena_mib is None:
                    selected_arena = probe_arena(
                        args, context_capacity, baseline.used_mib,
                        logs_dir, results_path,
                    )
                    measurement = benchmark_with_backoff(
                        args, context_capacity, selected_arena,
                        baseline.used_mib, logs_dir, results_path,
                    )
                else:
                    selected_arena = args.fixed_arena_mib
                    measurement = run_measurement(
                        args, context_capacity, selected_arena,
                        baseline.used_mib, logs_dir,
                    )
                append_jsonl(results_path, measurement)
                rows[context_capacity] = measurement
                write_csv(csv_path, rows)
                plot_results(args.output_dir, rows, plt)
            except Exception as exc:
                failed = True
                append_jsonl(
                    results_path,
                    {
                        "type": "point_failure",
                        "status": "failed",
                        "context_capacity": context_capacity,
                        "error": str(exc)[-4000:],
                    },
                )
                print(
                    "Stopping at the first failed context. Re-run with the "
                    "same --output-dir to resume after correcting the issue.",
                    file=sys.stderr,
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        interrupted = True
        print("Sweep interrupted; completed points remain resumable.", flush=True)
    finally:
        if rows:
            write_csv(csv_path, rows)
            plot_results(args.output_dir, rows, plt)

    print(f"JSONL: {results_path}", flush=True)
    if rows:
        print(f"CSV:   {csv_path}", flush=True)
        print(f"Plot:  {args.output_dir / 'kv-stream-sweep.png'}", flush=True)
    if interrupted:
        return 130
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
