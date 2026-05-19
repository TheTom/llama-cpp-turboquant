#!/bin/sh
source ../vulkan/1.4.350.0/setup-env.sh
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release -j $(nproc)
