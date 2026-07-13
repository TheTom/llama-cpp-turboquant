# OSCAR KV Cache — Baselines and Benchmarks

## Models

Pre-rotated Gemma 4 12B IT GGUFs used for baselines:

- Q4_K_M: `/mnt/storage/models/OSCAR/q4km-rot-kv/gemma-4-12b-it-rot-kv.gguf`
- Q8_0:   `/mnt/storage/models/google/gemma-4-12b-it-Q8_0-rot-kv.gguf`

## Important: No Rotation Overrides

These GGUFs have the DOA rotation baked into K/V during quantization.
**DO NOT** set `LLAMA_ATTN_ROT_K_OVERRIDE` or `LLAMA_ATTN_ROT_V_OVERRIDE` —
it would double-apply rotation and corrupt attention.

## CPU Baseline — q2_0 K + f16 V (Working)

```bash
CUDA_VISIBLE_DEVICES="" ./build/bin/llama-cli \
    -m /mnt/storage/models/OSCAR/q4km-rot-kv/gemma-4-12b-it-rot-kv.gguf \
    -ngl 0 -fa on -c 512 \
    --cache-type-k q2_0 --cache-type-v f16 \
    --chat-template-file models/templates/google-gemma-4-31B-it.jinja \
    -p "2+2=" -n 5 --temp 0
```

**Output:** `2+2=4` (correct)

## CPU Baseline — turbo2/turbo2 at 256k (Working)

```bash
CUDA_VISIBLE_DEVICES="" ./build/bin/llama-cli \
    -m /mnt/storage/models/google/gemma-4-12b-it-Q8_0-rot-kv.gguf \
    -ngl 0 -fa on -c 262144 \
    --cache-type-k turbo2 --cache-type-v turbo2 \
    --chat-template-file models/templates/google-gemma-4-31B-it.jinja \
    -p "What is the capital of France?" -n 80 --temp 0 --single-turn
```

**Output:** `The capital of France is **Paris**.` (correct)

**Speed:** Prompt 15.7 t/s, Generation 3.1 t/s

## GPU Blackwell (RTX 5090) Status

The turboquant VEC flash-attention kernel was broken on Blackwell (sm_120).
After replacing it with the mainline `fattn-vec.cuh` plus minimal q2_0/type
changes, flash attention works on the RTX 5090 for f16 and turbo2 KV types.
The custom q2_0 flash-attention kernel also now works after adding the missing
Hadamard transform to the write kernel and fixing mask indexing. q4_0 remains
broken on GPU.

Test command used for all GPU runs:

```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
    -m /mnt/storage/models/google/gemma-4-12b-it-Q8_0-rot-kv.gguf \
    -ngl 99 -fa on -c 262144 \
    --cache-type-k <KTYPE> --cache-type-v <VTYPE> \
    --chat-template-file models/templates/google-gemma-4-31B-it.jinja \
    -p "What is the capital of France?" -n 80 --temp 0 --single-turn
```

### f16/f16 — Working

```bash
--cache-type-k f16 --cache-type-v f16
```

**Output:** `The capital of France is Paris` (correct)

**Speed:** Prompt 693.5 t/s, Generation 95.5 t/s

### turbo2/turbo2 — Working

```bash
--cache-type-k turbo2 --cache-type-v turbo2
```

**Output:** `The capital of France is **Paris**.` (correct)

**Speed:** Prompt ~600 t/s, Generation ~88 t/s

**VRAM:** ~15,018 MiB (nvidia-smi dmon). This includes the Q8_0 model weights
and the KV-cache allocation.

### q4_0/q4_0 — Broken

```bash
--cache-type-k q4_0 --cache-type-v q4_0
```

**Output:** `<|channel>thought` repeated (garbage)

### q2_0/q2_0 — Working

```bash
--cache-type-k q2_0 --cache-type-v q2_0
```

**Output:** `The capital of France is Paris.` (correct)

**Speed:** Prompt ~8.8 t/s, Generation ~1.3 t/s

**VRAM:** ~14,778 MiB (nvidia-smi dmon)

Required fixes:
- Added missing Hadamard transform to the CUDA `set_rows` q2_0 write kernel
  (`q2_0_hadamard_inplace`) so quant/dequant are matched.
- Fixed mask indexing in the custom q2_0 flash-attention kernel for
  `ncols > 1` prompt processing (`maskh[j*ne11 + i_kv]`).

## HP Buffer Issues (All Architectures)

Adding the HP precision buffer changes the output even when the LP cache is
bit-exact relative to the HP cache:

```bash
CUDA_VISIBLE_DEVICES="" LLAMA_KV_FUSED_FA=1 \
LLAMA_KV_HP_SINK=64 LLAMA_KV_HP_RECENT=256 \
./build/bin/llama-cli \
    -m /mnt/storage/models/OSCAR/q4km-rot-kv/gemma-4-12b-it-rot-kv.gguf \
    -ngl 0 -fa on -c 512 \
    --cache-type-k q2_0 --cache-type-v f16 \
    --chat-template-file models/templates/google-gemma-4-31B-it.jinja \
    -p "2+2=" -n 5 --temp 0
```

**Output:** `2+2=2` (wrong)

Findings:
- Even with **all-f16 K+V** (`--cache-type-k f16 --cache-type-v f16`), the HP
  buffer still gives `2+2=2`, eliminating q2_0 quantization as the cause.
- `LLAMA_KV_CLIP_RATIO` (clipping) does not affect the result.
- `zero_hp_in_lp_mask` (double-counting prevention) does not change outcome.
- The concat softmax path (without `LLAMA_KV_FUSED_FA`) also gives a different
  wrong answer.

The HP buffer stores identical K/V data (same k_cur/v_cur) at f16 precision,
but the mixed FA kernel produces different attention values when combining LP
and HP tiers. Root cause unknown — likely a subtle data-layout issue in the HP
buffer view or a numerical interaction in the combined softmax.

GPU path with the HP buffer enabled also crashes in `set_input_hp_k_idxs`.

## Relevant Commits

- `5273f5672` — fix: use mainline fattn-vec.cuh
- `b289e4510` — fix: port minimal q2_0 VEC changes to mainline VEC kernel
- `9006f93a9` — fix: Q_ds loading index for nthreads_KQ_for_dot stride
- `88c3081ce` — q2_0 VEC kernel fixes (warp_reduce_sum, V stride, KQ guard)
- `e4537b059` — restore KQ_reg assignment
