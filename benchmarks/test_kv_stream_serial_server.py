#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def request(port: int, path: str, payload: dict | None, timeout: int = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


class Server:
    def __init__(
        self, model: Path, executable: Path, port: int,
        arena_mib: int, cache_mib: int, log: Path,
    ):
        command = [
            str(executable),
            "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", "8448", "-fa", "on", "-ctk", "q8_0", "-ctv", "q4_0",
            "-ngl", "all", "-b", "256", "-ub", "256", "-np", "1",
            "--no-mmproj", "--no-warmup", "--fit", "off",
            "--reasoning-format", "none",
            "--kv-stream-arena-mib", str(arena_mib),
            "--cache-ram", str(cache_mib),
        ]
        self.log_file = log.open("wb")
        self.process = subprocess.Popen(
            command, stdout=self.log_file, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log_file.close()
                raise RuntimeError(f"server exited with {self.process.returncode}")
            try:
                if request(port, "/health", None, 2).get("status") == "ok":
                    self.port = port
                    return
            except Exception:
                time.sleep(0.25)
        self.process.send_signal(signal.SIGINT)
        self.process.wait(timeout=15)
        self.log_file.close()
        raise RuntimeError("server did not become ready")

    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=15)
        self.log_file.close()


def completion(server: Server, prompt: list[int], cache_prompt: bool) -> dict:
    return request(server.port, "/completion", {
        "prompt": prompt,
        "n_predict": 16,
        "ignore_eos": True,
        "cache_prompt": cache_prompt,
        "temperature": 0,
        "seed": 1,
        "reasoning_format": "none",
    })


def patterned(size: int, tokens: tuple[int, ...]) -> list[int]:
    return [tokens[i % len(tokens)] for i in range(size)]


def run_serial(
    model: Path, executable: Path, port: int, arena_mib: int, output: Path,
):
    server = Server(
        model, executable, port, arena_mib, 0, output / "serial.log")
    try:
        short = patterned(1024, (23066, 1000, 2000))
        medium = patterned(4096, (23066, 3000, 4000, 5000))
        streamed = patterned(6144, (23066, 6000, 7000, 8000, 9000))
        short_first = completion(server, short, False)
        medium_first = completion(server, medium, False)
        streamed_result = completion(server, streamed, False)
        short_second = completion(server, short, False)
        medium_second = completion(server, medium, False)
        serial_changed = (
            short_second["content"] != short_first["content"] or
            medium_second["content"] != medium_first["content"])
        streamed_invalid = not streamed_result["content"].strip("/")
        if serial_changed or streamed_invalid:
            details = {
                "short_first": short_first["content"],
                "short_second": short_second["content"],
                "medium_first": medium_first["content"],
                "streamed": streamed_result["content"],
                "medium_second": medium_second["content"],
            }
            raise RuntimeError(f"serial unrelated-prefill output changed: {json.dumps(details)}")
        print("serial unrelated-prefill test: PASS", flush=True)
    finally:
        server.stop()


def run_prompt_cache(
    model: Path, executable: Path, port: int, arena_mib: int, output: Path,
):
    server = Server(
        model, executable, port, arena_mib, 1536,
        output / "prompt-cache.log")
    try:
        cached = patterned(4096, (23066, 1100, 2100, 3100))
        unrelated = patterned(2048, (23066, 4100, 5100, 6100))
        expected = completion(server, cached, True)["content"]
        completion(server, unrelated, True)
        restored = completion(server, cached, True)
        if restored["content"] != expected:
            raise RuntimeError("prompt-cache restore output changed")
        if restored["timings"].get("cache_n", 0) == 0:
            raise RuntimeError("prompt-cache restore did not reuse cached tokens")
        print(f"prompt-cache restore test: PASS (cache_n={restored['timings']['cache_n']})", flush=True)
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--server", type=Path,
        default=ROOT / "build-kv-cuda/bin/llama-server")
    parser.add_argument("--arena-mib", type=int, default=128)
    parser.add_argument("--port", type=int, default=12358)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/serial-server")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_serial(args.model, args.server, args.port, args.arena_mib, args.output)
    run_prompt_cache(
        args.model, args.server, args.port, args.arena_mib, args.output)


if __name__ == "__main__":
    main()
