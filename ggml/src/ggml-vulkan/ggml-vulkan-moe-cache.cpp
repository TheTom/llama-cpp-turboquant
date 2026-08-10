// Vulkan MoE Expert Cache — stub skeleton.
//
// To activate:
//   1. Add ggml-vulkan-moe-cache.cpp to ggml-vulkan CMakeLists.txt sources
//   2. Call ggml_vulkan_moe_cache_register() from ggml_backend_vk_reg()
//      in ggml-vulkan.cpp
//   3. Implement the TODO stubs below
//   4. Add GLSL compute shaders for expert matvec + fused SwiGLU
//
// Reference implementation: ggml-cuda/moe-cache.cu
// Shared data structures:   ggml-moe-cache-common.h

#include "../ggml-moe-cache-common.h"

// ---------------------------------------------------------------------------
// Vulkan-specific device extension
// ---------------------------------------------------------------------------

struct moe_cache_vulkan_device : public moe_cache_device {
    moe_cache_vulkan_device(int logical, int physical) : moe_cache_device(logical, physical) {}

    // TODO: VkDevice, VkQueue, VkCommandPool
    // TODO: VkBuffers for staging (h_input, h_out), device (d_input, d_act_q8, d_out)
    // TODO: VkPipeline / VkPipelineLayout for expert matvec and fused SwiGLU
    // TODO: VkDescriptorSetLayout / VkDescriptorPool / VkDescriptorSet
};

// ---------------------------------------------------------------------------
// API implementation stubs
// ---------------------------------------------------------------------------

static int vk_query_config(int automatic, size_t budget_mib, ggml_moe_cache_config * config) {
    (void) automatic;
    (void) budget_mib;
    (void) config;
    // TODO: populate config based on Vulkan device capabilities
    //  - min_expert_bytes based on subgroup size and cooperative matrix support
    //  - compute_capability equivalent (subgroup size, coopmat tier)
    return -1; // not supported yet
}

static int vk_query_device(void * device, const ggml_moe_cache_config * config,
                           ggml_moe_cache_device_caps * caps) {
    (void) device;
    (void) config;
    (void) caps;
    // TODO: return compute capability, min expert bytes, etc.
    return -1;
}

static int vk_query_shape(int wtype, int64_t n_in, int64_t n_out, int64_t n_expert,
                           size_t expert_size, ggml_moe_cache_shape_caps * caps) {
    (void) wtype;
    (void) n_in;
    (void) n_out;
    (void) n_expert;
    (void) expert_size;
    (void) caps;
    // TODO: return scratch/pool sizes for Vulkan matvec dispatch
    //  - staging buffer sizes for H2D/D2H copies
    //  - pool slab size
    return -1;
}

static void * vk_session_create(void * const * backends, int n_backends,
                                 const ggml_moe_cache_config * config) {
    (void) backends;
    (void) n_backends;
    (void) config;
    // TODO: create moe_cache_session + moe_cache_vulkan_device instances:
    //  - allocate staging buffers (host-visible) for H2D and D2H
    //  - allocate device-local buffers for d_input, d_act_q8, d_out
    //  - create command pools and descriptor sets
    //  - start worker threads for async fill
    return nullptr;
}

static void vk_session_destroy(void * session) {
    (void) session;
    // TODO: join worker threads, free Vulkan resources, destroy devices
}

static void vk_session_enter(void * session) {
    (void) session;
    // TODO: set thread-local scope
}

static void vk_session_leave(void * session) {
    (void) session;
    // TODO: clear thread-local scope
}

static void * vk_begin(const ggml_moe_cache_tensor_desc * tensor, int pool,
                        int64_t n_tokens, int n_rows, const int32_t * ids,
                        const float * const * act_rows, uint64_t * hit_mask) {
    (void) tensor;
    (void) pool;
    (void) n_tokens;
    (void) n_rows;
    (void) ids;
    (void) act_rows;
    (void) hit_mask;
    // TODO: check cache hits against Vulkan device pools, queue misses
    //  - for hits: record descriptor set updates pointing to cached slabs
    //  - for misses: enqueue staging copy jobs for async H2D fill
    return nullptr;
}

static int vk_plan(void * node) {
    (void) node;
    // TODO: prepare dispatch:
    //  - issue vkCmdCopyBuffer for missed expert rows (host -> device staging -> device local)
    //  - insert pipeline barriers (host write -> transfer -> compute shader read)
    //  - pin slots, record command buffers
    return 0;
}

static int vk_dispatch(void * node) {
    (void) node;
    // TODO: submit Vulkan command buffers with compute shader dispatches
    //  - expert matvec shader: weight (expert_size) × activation (n_in) -> partial output
    //  - use push constants or descriptor sets for per-row parameters
    //  - pipeline barriers between dispatches
    return 0;
}

static int vk_collect(void * node) {
    (void) node;
    // TODO: wait for Vulkan compute to finish (vkQueueWaitIdle or timeline semaphore)
    //  - copy results from device-local d_out to host-visible staging buffer
    //  - read back results, unpin slots
    return 0;
}

static void vk_end(void * node) {
    (void) node;
    // TODO: free node resources, reset command buffers, return to pool
}

static void * vk_fused_begin(const ggml_moe_cache_tensor_desc * up,
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
    // TODO: fused SwiGLU dispatch on Vulkan:
    //  - gate·up matvec -> SwiGLU activation -> single fused shader
    //  - saves one round-trip through global memory vs separate dispatches
    return nullptr;
}

static void vk_invalidate(void * session, const void * tensor_base) {
    (void) session;
    (void) tensor_base;
    // TODO: invalidate all cache slots for the given tensor across all devices
}

static int vk_trim(void * session, size_t target_bytes) {
    (void) session;
    (void) target_bytes;
    // TODO: evict LRU slots to reach target bytes
    //  - mark slots as free, return slabs to pool
    return 0;
}

static const ggml_moe_cache_api vk_moe_cache_api = {
    /* .owner          = */ &vk_moe_cache_api,
    /* .query_config   = */ vk_query_config,
    /* .query_device   = */ vk_query_device,
    /* .query_shape    = */ vk_query_shape,
    /* .session_create  = */ vk_session_create,
    /* .session_destroy = */ vk_session_destroy,
    /* .session_enter   = */ vk_session_enter,
    /* .session_leave   = */ vk_session_leave,
    /* .begin           = */ vk_begin,
    /* .plan            = */ vk_plan,
    /* .dispatch        = */ vk_dispatch,
    /* .collect         = */ vk_collect,
    /* .end             = */ vk_end,
    /* .fused_begin     = */ vk_fused_begin,
    /* .invalidate      = */ vk_invalidate,
    /* .trim            = */ vk_trim,
};

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

extern "C" void ggml_vulkan_moe_cache_register(void) {
    ggml_moe_cache_register(&vk_moe_cache_api);
}
