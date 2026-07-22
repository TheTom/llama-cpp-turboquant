# OSCAR2 Models

Collection: [OSCAR2 Models](https://huggingface.co/collections/jabbatheduck/oscar2-models-6a6004563e2dbe1dedd28750)

## Model

| Name | File | KV cache | Context |
|------|------|----------|---------|
| Qwen3.6-27B-Q5KXL-Hadamard | `qwen3.6-27b-q5kxl-hadamard.gguf` | oscar2/oscar2 | 131072 |

## OSCAR2 Source

This model requires an oscar2-capable build of `llama.cpp` from the TurboQuant oscar branch.

- Repo: https://github.com/giveen/llama-cpp-turboquant
- Branch: `oscar`

## Build Instructions

```bash
git clone -b oscar https://github.com/giveen/llama-cpp-turboquant.git
cd llama-cpp-turboquant
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build . --config Release -j $(nproc)
```

## Run with this Model

```bash
./build/bin/llama-cli \
  -m /path/to/qwen3.6-27b-q5kxl-hadamard.gguf \
  -ngl 99 -fa on -c 131072 \
  --cache-type-k oscar2 --cache-type-v oscar2 \
  -n 512 --temp 0
```

## Notes

- Requires NVIDIA Blackwell GPU with CUDA 13.3 or compatible.
- KV cache stores ~2.25 bits/weight with oscar2 quantization.
- Use `--no-jinja` if chat template parsing fails with very long prompts.
