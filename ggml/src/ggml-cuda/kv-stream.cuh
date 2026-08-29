#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

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
    // page_bytes/stage_slots/layer_count mirror the generic
    // ggml_backend_kv_stream_runtime_new_for_device_t plugin signature for
    // forward compatibility with a future chunked-streaming design - the
    // current CUDA implementation doesn't use them.
    size_t   pool_bytes  = 0;
    size_t   page_bytes  = 0;
    uint32_t stage_slots = 0;
    uint32_t layer_count = 0;
};

// Allocates the pinned host buffer type for `params.device`. Returns
// nullptr on any allocation failure - the caller should fall back to the
// ordinary non-streaming KV cache path.
ggml_backend_cuda_kv_stream_runtime_t ggml_backend_cuda_kv_stream_runtime_new(
        const ggml_backend_cuda_kv_stream_params & params);

// Same as above, but takes plain scalar arguments instead of the CUDA-specific
// params struct, and returns/takes an opaque void* handle - this is the shape
// resolved by llama-kv-cache.cpp through ggml_backend_reg_get_proc_address
// (registered under "ggml_backend_kv_stream_runtime_new_for_device" in
// ggml-cuda.cu), so that generic code never needs to include this header.
void * ggml_backend_cuda_kv_stream_runtime_new_for_device(
        int device, size_t pool_bytes, size_t page_bytes, uint32_t stage_slots, uint32_t layer_count);

void ggml_backend_cuda_kv_stream_runtime_free(ggml_backend_cuda_kv_stream_runtime_t runtime);

// The pinned host buffer type backing the authoritative KV cache tensors.
// The KV cache requests this instead of the ordinary CUDA device buffer
// type for a layer when streaming is enabled.
ggml_backend_buffer_type_t ggml_backend_cuda_kv_stream_buffer_type(
        ggml_backend_cuda_kv_stream_runtime_t runtime);
