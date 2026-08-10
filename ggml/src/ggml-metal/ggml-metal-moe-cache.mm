// Metal MoE Expert Cache — stub skeleton.
//
// To activate:
//   1. Add ggml-metal-moe-cache.mm to ggml-metal CMakeLists.txt sources
//   2. Call ggml_metal_moe_cache_register() from ggml_backend_metal_reg_init()
//      in ggml-metal.cpp
//   3. Implement the TODO stubs below
//
// Reference implementation: ggml-cuda/moe-cache.cu
// Shared data structures:   ggml-moe-cache-common.h

#include "../ggml-moe-cache-common.h"

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

// ---------------------------------------------------------------------------
// Metal-specific device extension
// ---------------------------------------------------------------------------

struct moe_cache_metal_device : public moe_cache_device {
    moe_cache_metal_device(int logical, int physical) : moe_cache_device(logical, physical) {}

    id<MTLDevice> mtl_device = nil;
    id<MTLCommandQueue> mtl_queue = nil;
    // TODO: MTLBuffers for d_input, d_act_q8, d_out
    // TODO: MTLComputePipelineStates for expert matvec and fused SwiGLU
};

// ---------------------------------------------------------------------------
// API implementation stubs
// ---------------------------------------------------------------------------

static int metal_query_config(int automatic, size_t budget_mib, ggml_moe_cache_config * config) {
    (void) automatic;
    (void) budget_mib;
    (void) config;
    // TODO: populate config based on Metal device capabilities
    return -1; // not supported yet
}

static int metal_query_device(void * device, const ggml_moe_cache_config * config,
                              ggml_moe_cache_device_caps * caps) {
    (void) device;
    (void) config;
    (void) caps;
    // TODO: return compute capability, min expert bytes, etc.
    return -1;
}

static int metal_query_shape(int wtype, int64_t n_in, int64_t n_out, int64_t n_expert,
                              size_t expert_size, ggml_moe_cache_shape_caps * caps) {
    (void) wtype;
    (void) n_in;
    (void) n_out;
    (void) n_expert;
    (void) expert_size;
    (void) caps;
    // TODO: return scratch/pool sizes for Metal matvec dispatch
    return -1;
}

static void * metal_session_create(void * const * backends, int n_backends,
                                    const ggml_moe_cache_config * config) {
    (void) backends;
    (void) n_backends;
    (void) config;
    // TODO: create moe_cache_session + moe_cache_metal_device instances,
    //       allocate MTLBuffers, start worker threads
    return nullptr;
}

static void metal_session_destroy(void * session) {
    (void) session;
    // TODO: join worker threads, free MTLBuffers, destroy devices
}

static void metal_session_enter(void * session) {
    (void) session;
    // TODO: set thread-local scope
}

static void metal_session_leave(void * session) {
    (void) session;
    // TODO: clear thread-local scope
}

static void * metal_begin(const ggml_moe_cache_tensor_desc * tensor, int pool,
                           int64_t n_tokens, int n_rows, const int32_t * ids,
                           const float * const * act_rows, uint64_t * hit_mask) {
    (void) tensor;
    (void) pool;
    (void) n_tokens;
    (void) n_rows;
    (void) ids;
    (void) act_rows;
    (void) hit_mask;
    // TODO: check cache hits, queue misses for async fill
    return nullptr;
}

static int metal_plan(void * node) {
    (void) node;
    // TODO: prepare dispatch — pin slots, issue H2D copies for misses
    return 0;
}

static int metal_dispatch(void * node) {
    (void) node;
    // TODO: launch Metal compute kernels for cached expert rows
    return 0;
}

static int metal_collect(void * node) {
    (void) node;
    // TODO: wait for Metal kernels, copy results out, unpin slots
    return 0;
}

static void metal_end(void * node) {
    (void) node;
    // TODO: free node resources
}

static void * metal_fused_begin(const ggml_moe_cache_tensor_desc * up,
                                 const ggml_moe_cache_tensor_desc * gate,
                                 int glu_op, float up_min, float up_max,
                                 float gate_min, float gate_max,
                                 const int32_t * ids, int n_rows, int64_t n_tokens,
                                 const float * const * act_rows, uint64_t * hit_mask) {
    (void) up;
    (void) gate;
    (void) glu_op;
    (void) up_min;
    (void) up_max;
    (void) gate_min;
    (void) gate_max;
    (void) ids;
    (void) n_rows;
    (void) n_tokens;
    (void) act_rows;
    (void) hit_mask;
    // TODO: fused SwiGLU dispatch on Metal
    return nullptr;
}

static void metal_invalidate(void * session, const void * tensor_base) {
    (void) session;
    (void) tensor_base;
    // TODO: invalidate all cache slots for the given tensor
}

static int metal_trim(void * session, size_t target_bytes) {
    (void) session;
    (void) target_bytes;
    // TODO: evict LRU slots to reach target
    return 0;
}

static const ggml_moe_cache_api metal_moe_cache_api = {
    /* .owner          = */ &metal_moe_cache_api,
    /* .query_config   = */ metal_query_config,
    /* .query_device   = */ metal_query_device,
    /* .query_shape    = */ metal_query_shape,
    /* .session_create  = */ metal_session_create,
    /* .session_destroy = */ metal_session_destroy,
    /* .session_enter   = */ metal_session_enter,
    /* .session_leave   = */ metal_session_leave,
    /* .begin           = */ metal_begin,
    /* .plan            = */ metal_plan,
    /* .dispatch        = */ metal_dispatch,
    /* .collect         = */ metal_collect,
    /* .end             = */ metal_end,
    /* .fused_begin     = */ metal_fused_begin,
    /* .invalidate      = */ metal_invalidate,
    /* .trim            = */ metal_trim,
};

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

extern "C" void ggml_metal_moe_cache_register(void) {
    ggml_moe_cache_register(&metal_moe_cache_api);
}
