#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>

// Experimental block-granular KV cache streaming (CUDA/HIP/MUSA execution
// backend - this file builds unmodified for all three, same as the rest of
// ggml-cuda).
//
// The authoritative KV cache lives in the pinned host buffer type exposed
// here (cudaHostAllocMapped): FlashAttention dereferences it directly via
// CUDA's unified virtual addressing, so no explicit device-side cache or
// copy is needed for correctness - just the buffer-type swap.

struct ggml_backend_cuda_kv_stream_runtime;
typedef ggml_backend_cuda_kv_stream_runtime * ggml_backend_cuda_kv_stream_runtime_t;

struct ggml_backend_cuda_kv_stream_params {
    int device = 0;
};

// Allocates the pinned host buffer type for `params.device`. Returns
// nullptr on any allocation failure - the caller should fall back to the
// ordinary non-streaming KV cache path.
ggml_backend_cuda_kv_stream_runtime_t ggml_backend_cuda_kv_stream_runtime_new(
        const ggml_backend_cuda_kv_stream_params & params);

void ggml_backend_cuda_kv_stream_runtime_free(ggml_backend_cuda_kv_stream_runtime_t runtime);

// The pinned host buffer type backing the authoritative KV cache tensors.
// The KV cache requests this instead of the ordinary CUDA device buffer
// type for a layer when streaming is enabled.
ggml_backend_buffer_type_t ggml_backend_cuda_kv_stream_buffer_type(
        ggml_backend_cuda_kv_stream_runtime_t runtime);
