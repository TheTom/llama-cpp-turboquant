#!/usr/bin/env python3
"""Unit tests for the automatic adaptive KV benchmark driver."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("benchmark_kv_stream.py")
SPEC = importlib.util.spec_from_file_location("benchmark_kv_stream", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkKvStreamTest(unittest.TestCase):
    def test_parse_token_count(self) -> None:
        self.assertEqual(BENCHMARK.parse_token_count("192K"), 192 * 1024)
        self.assertEqual(BENCHMARK.parse_token_count("262144"), 262144)
        with self.assertRaises(argparse.ArgumentTypeError):
            BENCHMARK.parse_token_count("bad")

    def test_parse_args_resolves_launched_paths(self) -> None:
        model = Path("models/model.gguf")
        server = Path("build/bin/llama-server")
        args = BENCHMARK.parse_args(
            [
                "--model", str(model),
                "--server", str(server),
                "--max-context", "8K",
                "--batch-size", "768",
                "--ubatch-size", "512",
            ]
        )
        self.assertEqual(args.model, model.resolve())
        self.assertEqual(args.server, server.resolve())
        self.assertEqual(args.batch_size, 768)
        self.assertEqual(args.ubatch_size, 512)
        self.assertEqual(args.probe_arena_mib, 1536)

    def test_validate_args_rejects_ubatch_larger_than_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            server = root / "llama-server"
            model.touch()
            server.touch(mode=0o755)
            args = BENCHMARK.parse_args(
                [
                    "--model", str(model),
                    "--server", str(server),
                    "--max-context", "8K",
                    "--batch-size", "256",
                    "--ubatch-size", "512",
                ]
            )
            with self.assertRaisesRegex(SystemExit, "must not exceed"):
                BENCHMARK.validate_args(args)

    def test_context_capacities_include_non_aligned_maximum(self) -> None:
        self.assertEqual(
            BENCHMARK.context_capacities(20000),
            [8192, 16384, 20000],
        )

    def test_context_capacities_honor_custom_start_and_step(self) -> None:
        self.assertEqual(
            BENCHMARK.context_capacities(163840, 40960, 8192),
            list(range(40960, 163841, 8192)),
        )


    def test_arena_estimate_uses_all_free_memory_and_rounds_down(self) -> None:
        self.assertEqual(BENCHMARK.estimate_arena_mib(1536, 2028, 32), 3552)
        self.assertEqual(
            BENCHMARK.estimate_arena_mib(1536, 2028, 32, max_arena_mib=2048),
            2048,
        )

    def test_clean_server_env_removes_memory_policy_overrides(self) -> None:
        inherited = {
            "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",
            "GGML_CUDA_PREFER_MODEL_WEIGHTS": "1",
            "GGML_CUDA_KV_STREAM_FIXED_RING_SLOTS": "8",
            "KEEP_ME": "yes",
        }
        with mock.patch.dict(os.environ, inherited, clear=True):
            env = BENCHMARK.clean_server_env("2")
        self.assertNotIn("GGML_CUDA_ENABLE_UNIFIED_MEMORY", env)
        self.assertNotIn("GGML_CUDA_PREFER_MODEL_WEIGHTS", env)
        self.assertNotIn("GGML_CUDA_KV_STREAM_FIXED_RING_SLOTS", env)
        self.assertEqual(env["KEEP_ME"], "yes")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")
        traced = BENCHMARK.clean_server_env(None, trace_kv_stream=True)
        self.assertEqual(traced["LLAMA_KV_STREAM_TRACE"], "1")

    def test_server_command_uses_tested_configuration(self) -> None:
        args = argparse.Namespace(
            server=Path("/tmp/llama-server"),
            model=Path("/tmp/model.gguf"),
            port=12355,
            extra_server_arg=["--verbosity", "3"],
            cache_type_k="bf16",
            cache_type_v="q8_0",
            batch_size=768,
            ubatch_size=512,
        )
        command = BENCHMARK.server_command(args, 131072, 2304)
        self.assertEqual(command[0], "/tmp/llama-server")
        self.assertIn("131072", command)
        self.assertIn("2304", command)
        self.assertEqual(command[command.index("-ctk") + 1], "bf16")
        self.assertEqual(command[command.index("-ctv") + 1], "q8_0")
        self.assertEqual(command[command.index("-b") + 1], "768")
        self.assertEqual(command[command.index("-ub") + 1], "512")
        self.assertEqual(command[command.index("-np") + 1], "1")
        self.assertEqual(
            command[command.index("--kv-stream-arena-mib") + 1], "2304"
        )
        self.assertEqual(command[command.index("--fit") + 1], "off")
        self.assertNotIn("--kv-stream-stage-mib", command)
        self.assertEqual(command[-2:], ["--verbosity", "3"])

    def test_prepare_server_warms_a_full_physical_batch(self) -> None:
        args = argparse.Namespace(
            prompt_suffix="suffix",
            fill_token_id=None,
            ubatch_size=512,
            request_timeout=30,
        )
        server = mock.Mock()
        server.url.side_effect = lambda path: f"http://test{path}"
        with mock.patch.object(
            BENCHMARK,
            "http_json",
            side_effect=({"tokens": [42, 43]}, {"timings": {}}),
        ) as http_json:
            suffix, fill_token_id = BENCHMARK.prepare_server(args, server)
        self.assertEqual(suffix, [42, 43])
        self.assertEqual(fill_token_id, 42)
        warmup_payload = http_json.call_args_list[1].args[1]
        self.assertEqual(len(warmup_payload["prompt"]), 512)

    def test_trace_parser_marks_only_pages_beyond_resident_partition(self) -> None:
        log = (
            "I kv_stream_adapt: active 65536, resident 256, ring 32, "
            "samples 1, misses 0, copy busy 0.0%, peak 1\n"
            "W kv_stream_adapt: adaptive KV partition: resident pages/layer "
            "256 -> 248, ring slots 32 -> 160, miss 50.0%, copy busy 25.0%\n"
            "I kv_stream_adapt: active 65792, resident 248, ring 160, "
            "samples 2, misses 1, copy busy 25.0%, peak 10\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(log)
            parsed = BENCHMARK.parse_kv_stream_trace(path)
        self.assertTrue(parsed["streaming_active"])
        self.assertEqual(parsed["stream_first_active_tokens"], 65792)
        self.assertEqual(parsed["stream_trace_samples"], 2)
        self.assertEqual(parsed["stream_repartitions"], 1)
        self.assertEqual(parsed["stream_max_ring_slots"], 160)


    def test_resume_rejects_changed_settings(self) -> None:

        signature = {"model": "/tmp/model.gguf", "max_context": 16384}
        BENCHMARK.validate_resume(signature.copy(), signature, Path("results.jsonl"))
        with self.assertRaisesRegex(SystemExit, "different settings"):
            BENCHMARK.validate_resume(
                {"model": "/tmp/other.gguf", "max_context": 16384},
                signature,
                Path("results.jsonl"),
            )

    def test_csv_and_plot_accept_partial_sweep(self) -> None:
        rows = {
            8192: {
                "context_capacity": 8192,
                "prompt_tokens": 7936,
                "decode_tokens": 256,
                "arena_mib": 3552,
                "prefill_tps": 1400.0,
                "decode_tps": 50.0,
            },
            16384: {
                "context_capacity": 16384,
                "prompt_tokens": 16128,
                "decode_tokens": 256,
                "arena_mib": 3520,
                "prefill_tps": 1300.0,
                "decode_tps": 45.0,
            },
        }
        try:
            plt = BENCHMARK.require_matplotlib()
        except SystemExit:
            self.skipTest("Matplotlib is not installed")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            BENCHMARK.write_csv(output / "results.csv", rows)
            BENCHMARK.plot_results(output, rows, plt)
            self.assertTrue((output / "results.csv").is_file())
            self.assertTrue((output / "kv-stream-sweep.png").is_file())
            self.assertTrue((output / "kv-stream-sweep.svg").is_file())


if __name__ == "__main__":
    unittest.main()
