# OSCAR INT2 KV Cache — CPU Benchmarks

Hardware: Intel Core Ultra 9 285K, 253 GB RAM
Model: Gemma-4-12B-It, Q8_0 weights, pre-rotated KV (oscar-rotation baked in)
Context: 262,144 tokens (256K)
Flash Attention: on (`-fa on`)

## Test Command

```bash
CUDA_VISIBLE_DEVICES="" ./build/bin/llama-cli \
    -m /mnt/storage/models/google/gemma-4-12b-it-Q8_0-rot-kv.gguf \
    -ngl 0 -c 262144 \
    --cache-type-k [TYPE] --cache-type-v [TYPE] \
    --chat-template-file models/templates/google-gemma-4-31B-it.jinja \
    -p "What is the capital of France?" -n 20 --temp 1.0 --single-turn
```

Where `[TYPE]` is `q4_0` or `q2_0`.

## Results

| Metric | Q4_0 K+V | Q2_0 K+V | Savings |
|---|---|---|---|
| Model weights | 13,097 MiB | 13,097 MiB | — |
| **KV cache** | **1,287 MiB** | **858 MiB** | **33% ↓** |
| Compute buffers | 371 MiB | 371 MiB | — |
| **Total memory** | **14,755 MiB** | **14,326 MiB** | **2.9% ↓** |
| Prompt speed | 13.9 t/s | 13.5 t/s | ~same |
| Generation speed | 2.7 t/s | 2.7 t/s | ~same |
| Output quality | "Paris is the capital of France" | "The capital of France is Paris." | Both correct |

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

## Observations

- KV cache at q2_0 uses **429 MiB less** than q4_0 at 256k context (33% reduction)
- Total system memory savings are modest (2.9%) because the Q8_0 model weights (~13 GB) dominate
- Generation speed is identical (~2.7 t/s) — CPU bottleneck is weight matmul, not KV cache bandwidth
- Output quality is preserved — both produce correct answers with the rotation

## Notes

- CPU-only test (`-ngl 0`). GPU path with q2_0 flash attention is still under development.
- GPU with f16 K+V and `-fa on` achieves **~682 t/s prompt / ~95 t/s generation** on RTX 5090.
- The Q8_0 model weights are ~13 GB regardless of cache format.
