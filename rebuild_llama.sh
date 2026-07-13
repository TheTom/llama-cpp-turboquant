#!/bin/bash
set -e 

LLAMA_DIR="/mnt/storage/Projects/turboquant"
BUILD_DIR="$LLAMA_DIR/build"
LOG_FILE="$LLAMA_DIR/rebuild.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "===================================================="
echo "LOCAL REBUILD STARTED: $(date)"
echo "===================================================="

error_handler() {
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: Script failed at line $1. Check $LOG_FILE for details."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "===================================================="
}
trap 'error_handler $LINENO' ERR

# Ensure PATH and LD_LIBRARY_PATH point to your 13.1 installation
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

if ! command -v nvcc &> /dev/null; then
    echo "--- Error: nvcc not found in PATH ---"
    exit 1
fi

cd "$LLAMA_DIR"

echo "--- Skipping update checks (Offline Mode) ---"
# Network pull removed to protect your current branch state.

echo "--- Verifying UI assets ---"
# We won't attempt to download anything. We just check if they exist.
UI_DIST_DIR="$LLAMA_DIR/tools/ui/dist"
if [ -d "$UI_DIST_DIR" ] && [ -n "$(ls -A "$UI_DIST_DIR" 2>/dev/null)" ]; then
    echo "  UI dist found at $UI_DIST_DIR. Proceeding."
else
    echo "  WARNING: No UI assets found at $UI_DIST_DIR."
    echo "  Build will proceed without an embedded UI."
fi

if [ -d "$BUILD_DIR" ]; then
    echo "--- Purging old CMake build cache ---"
    rm -rf "$BUILD_DIR"
fi

echo "--- Configuring CMake for RTX 5090 + Intel 285K ---"
# We lied about the compiler ID, so now we must provide the default standards manually
cmake -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_C_COMPILER=gcc-15 \
  -DCMAKE_CXX_COMPILER=g++-15 \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_FLAGS="-ccbin /usr/bin/g++-15 -isystem /usr/local/cuda/include" \
  -DCMAKE_CUDA_COMPILER_ID=NVIDIA \
  -DCMAKE_CUDA_COMPILER_VERSION=13.3 \
  -DCMAKE_CUDA_STANDARD_COMPUTED_DEFAULT=17 \
  -DCMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DGGML_LTO=ON \
  -DGGML_CPU_KLEIDIAI=OFF \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_LINK_DEPENDS_USE_LINKER=OFF

echo "--- Starting the build ---"
cmake --build "$BUILD_DIR" --config Release -j "$(nproc)"

echo "--- Build complete ---"
echo "===================================================="
echo "REBUILD COMPLETED SUCCESSFULLY: $(date)"
echo "===================================================="
