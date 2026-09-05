# Adaptive KV streaming benchmark

`benchmark_kv_stream.py` performs the complete context-matched benchmark and
creates its graph. The only required inputs are the GGUF model and the largest
context capacity to test:

```bash
python3 benchmarks/benchmark_kv_stream.py \
  --model /path/to/model.gguf \
  --max-context 192K
```

The default server is `build/bin/llama-server`. Use `--server` when the
binary is elsewhere. Matplotlib is the only Python dependency:

```bash
python3 -m pip install matplotlib
```

## What the script does

For every configured context capacity from 8K through `--max-context`, in 8K
steps, the script:

1. Starts a fresh adaptive KV streaming server with a 1536 MiB total arena.
2. If that arena cannot fit the prefill graph plus minimum KV layout, increases it in 256 MiB steps and retries.
3. Measures free VRAM after model initialization and a warm-up request.
4. Adds all measured free VRAM to the successful probe arena, rounded down to 32 MiB.
5. Runs a full prompt and 256-token decode to validate that arena.
6. If the candidate fails, reduces it by 64 MiB and retries.
7. Records prefill speed, decode speed, selected arena size, and VRAM telemetry.
8. Updates the CSV and Matplotlib graph after every successful point.

There is no fixed VRAM safety reserve. Actual server execution is the
validation: allocation failures are handled by automatic arena backoff.

The arena is one physical CUDA allocation shared between resident/ring KV and
the active phase's compute workspace. The configured MiB value therefore
includes both. Prompt processing reserves its measured scheduler workspace;
TG1 decode releases that large slice and gives the reclaimed bytes to KV.

The prompt length at each point is the configured context capacity minus the
256 decode tokens. For example, the 192K point starts the server with
`--ctx-size 196608`, prefills 196352 tokens, and then decodes 256 tokens.
If the maximum is not a multiple of 8K, the exact maximum is appended as the
last point.

The driver uses the configuration currently supported and validated by this
branch:

- Flash Attention enabled
- K cache `q8_0`
- V cache `q4_0`
- all model layers on the GPU
- one server slot
- 256-token batch and micro-batch by default
- ordinary CUDA allocation, without UVM

Use `--batch-size` and `--ubatch-size` to benchmark other logical and physical batch sizes. The micro-batch must not exceed the logical batch. Both values are included in result metadata and the resume signature.

```bash
python3 benchmarks/benchmark_kv_stream.py \
  --model /path/to/model.gguf \
  --max-context 192K \
  --batch-size 512 \
  --ubatch-size 512
```

## Results and resuming

By default, a timestamped directory is created under
`benchmarks/results/adaptive-kv-sweep-*`. It contains:

- `results.jsonl`: metadata, arena probes, retries, and measurements
- `results.csv`: one successful measurement per context capacity
- `kv-stream-sweep.png` and `kv-stream-sweep.svg`: decode, prefill, and arena
  size plots
- `logs/`: one server log per probe and benchmark attempt

Use an explicit output directory to resume an interrupted sweep:

```bash
python3 benchmarks/benchmark_kv_stream.py \
  --model /path/to/model.gguf \
  --max-context 192K \
  --output-dir benchmarks/results/my-sweep
```

Re-run the same command after an interruption. Completed contexts are skipped.
The script rejects a resume if the model or benchmark settings differ, avoiding
mixed data in one result set.

Run `python3 benchmarks/benchmark_kv_stream.py --help` for optional GPU,
timeout, arena probing/backoff, output, and server arguments. Old `--*-pool-*`
driver spellings remain accepted as compatibility aliases.

Do not run another GPU workload during the sweep. Its allocations would change
the automatically selected arena and invalidate comparisons between points.
