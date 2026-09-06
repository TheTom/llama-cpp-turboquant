# KV streaming benchmark raw results

Backing data for this PR's description. Referenced from the PR body as
`results.csv` under each benchmark's own name.

- `images/` - the five benchmark charts embedded in the PR description, at
  full resolution.
- `qwen3.8-27b-throughput/` - `benchmark_kv_stream.py`, Qwen3.8-27B-AD, 8K->262K, no MTP.
- `qwen3.8-27b-mtp-throughput/` - same, with `--spec-type draft-mtp --spec-chain 8`, 8K->106K.
- `gemma4-it-throughput/` - same, Gemma-4-26B-A4B-It, 8K->131K.
- `qwen3.8-27b-memory/` - `benchmark_kv_memory.py`, Qwen3.8-27B-AD, 131K->786K (YaRN-scaled 4x).
  `logs/baseline-393216.log` is the full gdb-style backtrace of the real
  CUDA OOM abort the no-streaming leg hit at 393K context.
- `gemma4-it-memory/` - same, Gemma-4-26B-A4B-It, 65K->393K (no-streaming never OOM'd in this range).
- `kld-logs/` - raw `llama-perplexity` output for every PPL/KL-divergence
  check in the PR draft's "Correctness" sections: baseline and streaming
  runs at 4K and 32K context for both models, the Gemma4 turbo/f16/no-FA
  isolation checks, the Gemma4 coherence spot-check, and the 131K attempt
  that hit a tooling limitation in `llama-perplexity`'s own KL-divergence
  code (`qwen-streaming-131k-FAILED-tooling-limitation.log`).

Not included: the saved-logits `.kld` files themselves (13-27 GB each,
regenerate with `--save-all-logits` if needed) and the per-point server
logs from the memory/throughput sweeps other than the one OOM backtrace
above (available in the original `--output-dir` if still on disk).
