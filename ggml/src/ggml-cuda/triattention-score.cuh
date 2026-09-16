/*
 * TriAttention GPU scoring kernel (CUDA)
 *
 * Computes TriAttention importance scores for KV cache entries directly on
 * the GPU, avoiding the costly GPU→CPU transfer of the full K tensor.
 * Only the resulting score array (one float per position) is copied back.
 *
 * Supports TurboQuant types (turbo2_0, turbo3_0, turbo4_0), Q8_0, F16, F32.
 *
 * Reference: "TriAttention: Trigonometric KV Cache Eviction" (arXiv 2604.04921)
 */

#pragma once

#include "common.cuh"

// TriAttention GPU scoring API declarations are in ggml-cuda.h (included via common.cuh).
// Only the CUDA kernel implementation details are declared here.
