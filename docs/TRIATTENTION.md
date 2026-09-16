# TriAttention: Trigonometric KV Cache Eviction for TurboQuant

## Overview

TriAttention is a **calibration-guided KV cache eviction** system that uses
trigonometric scoring to determine which tokens to keep in the KV cache during
long-context inference. It is based on the paper
["TriAttention: Trigonometric KV Cache Eviction"](https://arxiv.org/abs/2604.04921)
(MIT/NVIDIA/ZJU, 2025).

When combined with TurboQuant's 2-4 bit KV compression, TriAttention provides
**~40× effective KV memory reduction** (compression × eviction) while
maintaining quality.

## How it works

1. **Calibration** (offline): Run a short calibration pass over ~2K tokens to
   collect pre-RoPE query statistics per (layer, head, frequency-band).

2. **Scoring** (at runtime): When the KV cache fills beyond a budget threshold,
   TriAttention scores every cached token by asking: *"If future queries follow
   the calibration distribution, how much attention would this token receive?"*

3. **Eviction**: The lowest-scoring tokens are evicted to bring the cache back
   to the budget. Prompt tokens can optionally be protected.

### Scoring formula

For each cached key k at position m, the score integrates over geometric
time offsets (d) and frequency bands (f):

```
S(m) = (1/D) Σ_d Σ_f  amp_f · fscale²_f · cos(ω_f · δ_d + φ_f)  +  excess_f · fscale²_f · |k_f|
```

Where:
- `amp_f = ||E[q_f]|| · |k_f|` — amplitude from calibrated query mean × key magnitude
- `φ_f = angle(E[q_f] · conj(k_f))` — phase alignment between calibrated Q and cached K
- `ω_f = θ^(-2f/d)` — RoPE frequency for band f
- `δ_d = current_pos - key_pos + offset_d` — time offset (geometric progression)
- `fscale²_f` — frequency importance weighting (1/ω²)
- `excess_f = E[||q_f||] - ||E[q_f]||` — norm excess (measures query variance)

## Quick start

### 1. Generate calibration file

```bash
pip install torch transformers datasets tqdm numpy

python scripts/calibrate-triattention.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --n-tokens 2048 \
    --output models/llama3.1-8b.triattention
```

### 2. Run llama-server with TriAttention

```bash
./llama-server \
    -m models/llama-3.1-8b-turbo3.gguf \
    --triattention-stats models/llama3.1-8b.triattention \
    --triattention-budget 2048 \
    --triattention-window 128 \
    -c 131072 \
    --port 8090
```

### 3. Validate calibration file

```bash
python scripts/validate-calibration.py models/llama3.1-8b.triattention
python scripts/validate-calibration.py models/llama3.1-8b.triattention \
    --model meta-llama/Llama-3.1-8B-Instruct
```

## CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--triattention-stats PATH` | — | Path to `.triattention` calibration file (enables eviction) |
| `--triattention-budget N` | 2048 | Max KV entries retained after pruning |
| `--triattention-window N` | 128 | Pruning interval in decode tokens |
| `--triattention-offset-max N` | 65536 | Max geometric offset for scoring (controls time horizon) |
| `--triattention-mode MODE` | global | `global`, `per-kv-head`, or `per-layer-head` |
| `--triattention-trigger MODE` | interval | `interval` or `slack` |
| `--triattention-agg MODE` | mean | `mean` or `max` aggregation over offsets |
| `--triattention-seed N` | 0 | RNG seed for score tie-breaking (0=deterministic) |
| `--triattention-normalize` | off | Z-score normalize scores per head before selection |
| `--triattention-no-protect-prefill` | off | Allow eviction of prompt tokens |
| `--triattention-disable-mlr` | off | Ablation: disable MLR weighting |
| `--triattention-disable-trig` | off | Ablation: norm-only scoring |
| `--triattention-log` | off | Log pruning events to stderr |

## Pruning modes

### Global (default)
All sampled heads contribute to a single global importance score per position.
The top-B positions (by max-over-heads score) are retained. Best for most
use cases.

### Per-KV-head
Each KV head independently selects its own top-B tokens. Different heads may
retain different subsets. Can improve quality for GQA models where different
KV heads serve different query groups.

### Per-layer-head
Most fine-grained: each (layer, KV-head) pair selects independently. Highest
potential quality but highest overhead.

## Trigger strategies

### Interval (default)
Prune every `window` decode tokens when the cache occupancy exceeds the budget.
Simple and predictable.

### Slack
Prune when cache occupancy reaches `budget + window` (the slack threshold).
Adapts naturally to variable-length sequences and batch processing.

## Calibration file format

Binary format with magic `0x54524941` ("TRIA"), version 1. See the full
specification in `src/llama-triattention.h`.

```
Header (variable length):
  magic          u32    0x54524941
  version        u32    1
  head_dim       u32    e.g. 128
  num_layers     u32
  num_attn_heads u32    total attention heads
  num_kv_heads   u32    GQA KV heads
  rope_theta     f64
  rope_style     u32    0=half, 1=interleaved
  n_sampled      u32    number of calibrated head entries
  freq_count     u32    head_dim / 2
  name_len       u32
  name           char[name_len]

Per head (repeated n_sampled times):
  layer_idx      u32
  head_idx       u32
  q_mean_real    f32[freq_count]
  q_mean_imag    f32[freq_count]
  q_abs_mean     f32[freq_count]
  r_f            f32[freq_count]   (validation: ||E[q]||/E[||q||])
```

## Architecture

### Source files

| File | Purpose |
|------|---------|
| `src/llama-triattention.h` | Public API header — structs, enums, function declarations |
| `src/llama-triattention.cpp` | CPU implementation — scoring, pruning, calibration loading |
| `ggml/src/ggml-cuda/triattention-score.cu` | CUDA scoring kernel — GPU-accelerated importance scoring |
| `ggml/src/ggml-cuda/triattention-score.cuh` | CUDA kernel header |
| `scripts/calibrate-triattention.py` | Calibration script — collects Q stats from HF models |
| `scripts/validate-calibration.py` | Validation script — reads and checks `.triattention` files |

### Integration points

1. **KV cache hooks** (`src/llama-kv-cache.cpp`):
   - `apply_ubatch()` — tracks token additions and triggers pruning
   - `seq_rm()` / `clear()` — notifies position tracker of evictions
   - `seq_add()` — updates positions on cache shifts

2. **CLI registration** (`common/arg.cpp`):
   All 13 `--triattention-*` arguments registered for SERVER and CLI examples

3. **Public API** (`include/llama.h`):
   `llama_triattention_init()` — initializes TriAttention on a context

4. **Auto-init** (`common/common.cpp`):
   Automatically calls `llama_triattention_init()` after context creation when
   `--triattention-stats` is provided

### Data flow

```
                    ┌─────────────────────────┐
                    │  .triattention file      │
                    │  (calibration stats)     │
                    └────────┬────────────────┘
                             │ load at init
                             ▼
┌──────────┐    ┌─────────────────────────┐    ┌──────────────┐
│KV cache  │◄──►│  triattention_state     │    │ CUDA kernel  │
│ hooks    │    │  - calibration data     │───►│ (optional    │
│ (track   │    │  - omega, freq_scale    │    │  GPU accel)  │
│  cells)  │    │  - cell positions       │    └──────────────┘
└──────────┘    │  - scoring buffers      │
                └────────┬────────────────┘
                         │ prune
                         ▼
                ┌─────────────────────────┐
                │  triattention_prune_impl│
                │  1. Enumerate occupied  │
                │  2. Score per head      │
                │  3. Combine scores      │
                │  4. Select top-B        │
                │  5. Evict losers        │
                └─────────────────────────┘
```

## Performance characteristics

- **Scoring overhead**: ~5-10ms per pruning event (CPU path) for 128K context
  with TurboQuant types. GPU path: <1ms.
- **Memory overhead**: ~4 bytes per KV position (position tracking) + calibration
  data (~100KB for typical models)
- **Pruning frequency**: Every `window` tokens (default 128), or when slack
  threshold exceeded

## Compatibility

- **Quantization types**: TURBO2_0, TURBO3_0, TURBO4_0, Q8_0, F16, BF16, F32
- **RoPE styles**: Half layout (LLaMA, Mistral, Qwen2) and interleaved (GPT-NeoX)
- **GQA**: Full support — calibration covers all attention heads, scoring
  samples representative heads per KV group
- **Architectures**: Any transformer with standard RoPE-based attention

## References

- Paper: ["TriAttention: Trigonometric KV Cache Eviction"](https://arxiv.org/abs/2604.04921)
  Yaniv Ben-Nun, Agustín Zanotti, Dan Alistarh, Ming-Yu Liu (MIT, NVIDIA, ZJU)
- TurboQuant: ["TurboQuant: Online Vector-Level LLM KV-Cache Compression"](https://arxiv.org/abs/2504.19874)
  (ICLR 2026)
