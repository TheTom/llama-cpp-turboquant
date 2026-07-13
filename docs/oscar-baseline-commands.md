# OSCAR2 KV Cache — Baselines

## Model

Pre-rotated Gemma 4 12B IT, Q8_0:
`/mnt/storage/models/google/gemma-4-12b-it-Q8_0-rot-kv.gguf`

## Device

NVIDIA GeForce RTX 5090 (Blackwell, sm_120), CUDA 13.3

---

## Working Baselines

### f16/f16
```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
    -m /mnt/storage/models/google/gemma-4-12b-it-Q8_0-rot-kv.gguf \
    -ngl 99 -fa on -c 65536 \
    --cache-type-k f16 --cache-type-v f16 \
    --temp 0 -p "What is the capital of France?" -n 20 --single-turn
```
Output: `The capital of France is Paris`
Speed: Prompt ~700 t/s, Generation ~96 t/s

### q2_0/q2_0 (dedicated kernel)
```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
    -m /mnt/storage/models/google/gemma-4-12b-it-Q8_0-rot-kv.gguf \
    -ngl 99 -fa on -c 65536 \
    --cache-type-k q2_0 --cache-type-v q2_0 \
    --temp 0 -p "What is the capital of France?" -n 20 --single-turn
```
Output: `The capital of France is Paris`
Speed: Prompt ~135 t/s, Generation ~15 t/s

---

## Under Development

### oscar2/oscar2 (dedicated FA kernel)
```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
    -m /mnt/storage/models/google/gemma-4-12b-it-Q8_0-rot-kv.gguf \
    -ngl 99 -fa on -c 65536 \
    --cache-type-k oscar2 --cache-type-v oscar2 \
    --temp 0 -p "What is the capital of France?" -n 20 --single-turn
```
Output: Garbled (kernel bug)
Speed: Prompt ~265 t/s, Generation ~29 t/s
Status: Cache compresses correctly; decode kernel produces wrong attention.

### oscar2/oscar2 (VEC fallback path)
Same command but dispatches through the generic VEC kernel.
Output: Garbled (VEC path broken for quantized KV at D>256)
Speed: Prompt ~520 t/s, Generation ~86 t/s

---

## Test Commands (Round-Trip Verification)

### SET_ROWS + CPY (quantize + dequantize)
```bash
CUDA_VISIBLE_DEVICES=0 ./build/bin/test-backend-ops -o CPY 2>&1 | grep oscar2
```
(Test cases for oscar2 need to be added to test-backend-ops.cpp)
