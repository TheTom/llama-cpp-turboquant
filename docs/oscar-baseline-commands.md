# OSCAR q2_0 KV Cache — Working Baselines

## Model
Pre-rotated Gemma 4 12B IT GGUF:
`/mnt/storage/models/OSCAR/q4km-rot-kv/gemma-4-12b-it-rot-kv.gguf`

## Important: No Rotation Overrides

This GGUF has the DOA rotation baked into K/V during quantization.
**DO NOT** set `LLAMA_ATTN_ROT_K_OVERRIDE` or `LLAMA_ATTN_ROT_V_OVERRIDE` —
it would double-apply rotation and corrupt attention.

## CPU Baseline — Working

q2_0 K + f16 V, no HP buffer, no rotation, with chat template:

```bash
CUDA_VISIBLE_DEVICES="" ./build/bin/llama-cli \
    -m /mnt/storage/models/OSCAR/q4km-rot-kv/gemma-4-12b-it-rot-kv.gguf \
    -ngl 0 -fa on -c 512 \
    --cache-type-k q2_0 --cache-type-v f16 \
    --chat-template-file models/templates/google-gemma-4-31B-it.jinja \
    -p "2+2=" -n 5 --temp 0
```

**Output:** `2+2=4` (correct)

## HP Buffer Degrades Output

Adding the HP precision buffer changes the output:

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

### Investigation Results

- Even with **all-f16 K+V** (`--cache-type-k f16 --cache-type-v f16`), HP buffer still gives `2+2=2`
- Eliminates q2_0 quantization as the cause
- `LLAMA_KV_CLIP_RATIO` (clipping) doesn't affect the result
- `zero_hp_in_lp_mask` (double-counting prevention) doesn't change outcome
- The concat softmax path (without `LLAMA_KV_FUSED_FA`) also gives different wrong answer

The HP buffer stores identical K/V data (same k_cur/v_cur) at f16 precision. The
mixed FA kernel processes both LP and HP tiers identically. The root cause is
unknown — likely a subtle data layout issue in the HP buffer view or a numerical
interaction in the combined LP+HP softmax.

## GPU VEC Kernel Crash

GPU path with `-ngl 99` crashes (segfault) in `set_input_hp_k_idxs` when HP
buffer is enabled. Without HP buffer it runs but produces `<unused49>` because
q2_0 is too lossy for D=512 heads without the HP precision backup.

GPU VEC kernel fixes applied (committed `88c3081ce`):
1. warp_reduce_sum — uses nthreads_KQ_for_dot (8) not nthreads_KQ (32) for q2_0
2. V loop position stride — non-turbo V uses proper per-position stride
3. KQ garbage guard — zeros out-of-range KQ reads

KQ_reg assignment restored (committed `e4537b059`) — critical upstream line
that was accidentally dropped.

## GPU Blackwell (RTX 5090) Status

**Root cause: flash attention broken on Blackwell (sm_120).**

Both VEC and MMA flash attention kernels produce incorrect attention values
on RTX 5090 with CUDA 13.3. Diagnosed via:
- VulnLLM-R-7B (D=128): MMA produces garbled output ("statementnot");
  q2_0 VEC produces all-'2' repetition
- Gemma-4 12B (D=256/512): VEC/MMA both produce `<|channel>` garbage

**Fix applied** (commit `aa3256799`): flash attention returns NONE on Blackwell,
falling back to standard matmul (identical to `-fa off`).

### Working GPU config (Blackwell):

```bash
./build/bin/llama-cli \
    -m /mnt/storage/models/google/gemma-4-12b-it-Q8_0-rot-kv.gguf \
    -ngl 99 -fa off -c 1024 \
    --cache-type-k f16 --cache-type-v f16 \
    --chat-template-file models/templates/google-gemma-4-31B-it.jinja \
    -p "What is 2+2?" -n 100 --temp 0 --single-turn
```

**Output:** `2 + 2 = 4` at ~90 tok/s generation.

### Unresolved (needs upstream NVIDIA/CUDA fix):
- Flash attention (VEC/MMA) on Blackwell sm_120
- HP buffer precision degradation on all architectures
- GPU HP buffer segfault (`set_input_hp_k_idxs`)
