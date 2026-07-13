# OSCAR INT2 KV Cache — CPU Benchmarks

Hardware: Intel Core Ultra 9 285K, 253 GB RAM
Model: Gemma-4-12B-It, Q8_0 weights, pre-rotated KV (oscar-rotation baked in)
Context: 262,144 tokens (256K)  
Flash Attention: on (`-fa on`)  
WARP_SIZE: 32 (RTX 5090 Blackwell sm_120)

## Test Command

```bash
CUDA_VISIBLE_DEVICES="" ./build/bin/llama-cli \
    -m /mnt/storage/models/google/gemma-4-12b-it-Q8_0-rot-kv.gguf \
    -ngl 0 -c 262144 \
    --cache-type-k [TYPE] --cache-type-v [TYPE] \
    --chat-template-file models/templates/google-gemma-4-31B-it.jinja \
    -p "What is the capital of France?" -n 20 --temp 1.0 --single-turn
```

Where `[TYPE]` is `q4_0`, `q2_0`, or `turbo2`.

## Results

| Metric | Q4_0 K+V | Q2_0 K+V | Turbo2 K+V | Savings (turbo2 vs q4_0) |
|---|---|---|---|---|
| Model weights | 13,097 MiB | 13,097 MiB | 13,097 MiB | — |
| **KV cache** | **1,287 MiB** | **858 MiB** | **717 MiB** | **44% ↓** |
| Compute buffers | 371 MiB | 371 MiB | 379 MiB | — |
| **Total memory** | **14,755 MiB** | **14,326 MiB** | **14,193 MiB** | **3.8% ↓** |
| Prompt speed | 13.9 t/s | 13.5 t/s | 15.8 t/s | +14% |
| Generation speed | 2.7 t/s | 2.7 t/s | 3.1 t/s | +15% |
| Output quality | "Paris is the capital of France" | "The capital of France is Paris." | "The capital of France is Paris." | All correct |

## Memory Breakdown (verbose)

### Q4_0

```
| memory breakdown [MiB] | total   free     self   model   context   compute    unaccounted |
|   - Host               |                 14755 = 13097 +    1287 +     371                |
```

### Q2_0

```
| memory breakdown [MiB] | total   free     self   model   context   compute    unaccounted |
|   - Host               |                 14326 = 13097 +     858 +     371                |
```

### Turbo2

```
| memory breakdown [MiB] | total   free     self   model   context   compute    unaccounted |
|   - Host               |                 14193 = 13097 +     717 +     379                |
```

## Observations

- Turbo2 KV cache saves **570 MiB** vs q4_0 (44% reduction) and **141 MiB** vs q2_0 (16% further reduction) at 256k context
- Turbo2 uses a single half-precision scale per 32-element block (10 bytes/block) vs q2_0's two-scales (12 bytes/block), same 2-bit index storage
- Turbo2 shows a measurable speed advantage: **15.8 / 3.1 t/s** vs ~13.7 / 2.7 t/s for q4_0/q2_0 — likely from reduced KV cache memory bandwidth
- Total system memory savings are modest (3.8%) because the Q8_0 model weights (~13 GB) dominate
- Output quality is preserved across all three formats

## Notes

- CPU-only test (`-ngl 0`). GPU path with q2_0 flash attention is still under development.
- GPU with f16 K+V and `-fa on` achieves **~682 t/s prompt / ~95 t/s generation** on RTX 5090.
- The Q8_0 model weights are ~13 GB regardless of cache format.
- Turbo2 block size: 32 elements, 10 bytes/block (2 bytes norm + 8 bytes 2-bit indices). Effective rate: 2.5 bits/element.
- Q2_0 block size: 32 elements, 12 bytes/block (4 bytes scales + 8 bytes 2-bit indices). Effective rate: 3 bits/element.
