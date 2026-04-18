#!/usr/bin/env bash
set -euo pipefail

# Keep shader cache in a user-writable location.
MESA_SHADER_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/llama-mesa-cache"
mkdir -p "$MESA_SHADER_CACHE_DIR"

MESA_SHADER_CACHE_DIR="$MESA_SHADER_CACHE_DIR" \
./build-vulkan-only-2/bin/llama-server \
  -m "/run/media/marco/8a876936-c186-4375-b2d3-e191161a7fd6/@llama-models/Qwen_Qwen3.6-35B-A3B-Q3_K_XL.gguf" \
  -ngl 999 \
  -c 98304 \
  -ctk turbo3 \
  -ctv turbo3 \
  -np 1 \
  -b 2048 \
  -ub 512 \
  -fa on \
  --cache-reuse 0 \
  --metrics \
  --reasoning-budget 0
