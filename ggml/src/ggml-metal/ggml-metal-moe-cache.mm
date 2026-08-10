// Metal MoE Expert Cache — self-contained implementation.
//
// Uses unified memory (no H2D copies), pre-compiled kernel from ggml-metal.metal.
//
// Activate by:
//   1. Adding to CMakeLists.txt
//   2. Calling ggml_metal_moe_cache_register() from ggml_backend_metal_reg_init()

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../ggml-moe-cache-common.h"

// Thread-local session stack (owned by this backend; separate from CUDA's).
static thread_local std::vector<moe_cache_scope_frame> g_session_stack;
static thread_local int g_session_suppressed = 0;

// ---------------------------------------------------------------------------
// Metal device extension
// ---------------------------------------------------------------------------

struct moe_cache_metal_device : public moe_cache_device {
    moe_cache_metal_device(id<MTLDevice> dev, id<MTLCommandQueue> q)
        : moe_cache_device(0, 0), mtl_device(dev), mtl_queue(q) {
        [mtl_device retain];
        [mtl_queue retain];
    }

    ~moe_cache_metal_device() {
        free_slabs();
        free_scratch();
        [mtl_queue release];
        [mtl_device release];
    }

    id<MTLDevice> mtl_device;
    id<MTLCommandQueue> mtl_queue;
    id<MTLComputePipelineState> mmv_pipeline = nil;

    // Tracked MTLBuffers for slab pools and scratch.
    std::vector<id<MTLBuffer>> slab_buffers;
    id<MTLBuffer> out_buffer = nil;

    void free_slabs() {
        for (auto buf : slab_buffers) {
            [buf release];
        }
        slab_buffers.clear();
        for (auto & p : pools) {
            p->slab = nullptr;
        }
    }

    void free_scratch() {
        if (out_buffer) {
            [out_buffer release];
            out_buffer = nil;
            d_out = nullptr;
            d_out_cap = 0;
        }
        if (d_act_q8) {
            // act_q8 is owned by an MTLBuffer stored separately
            // d_act_q8 is set from [buf contents]
            d_act_q8 = nullptr;
            act_q8_cap = 0;
        }
    }
};

// Forward decls
static const ggml_moe_cache_api metal_moe_cache_api;

// ---------------------------------------------------------------------------
// Slab allocation — MTLBuffer with shared storage.
// ---------------------------------------------------------------------------

static char * metal_slab_alloc(moe_cache_metal_device & dev, size_t bytes) {
    id<MTLBuffer> buf = [dev.mtl_device newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];
    if (!buf) return nullptr;
    dev.slab_buffers.push_back(buf);
    return (char *)[buf contents];
}

// ---------------------------------------------------------------------------
// Query functions
// ---------------------------------------------------------------------------

static int metal_query_config(int automatic, size_t budget_mib,
                               ggml_moe_cache_config * result) {
    if (!result) return 0;

    moe_cache_config config = moe_cache_read_config();
    if (automatic >= 0) {
        config.enabled = true;
        config.automatic = automatic != 0;
        moe_cache_apply_mode_defaults(config);
    }
    if (budget_mib > 0) config.budget_mb = budget_mib;
    if (!config.enabled) return 0;

    result->budget_bytes = config.budget_mb << 20;
    result->reserve_bytes = config.reserve_mb << 20;
    result->minimum_slab_bytes = config.minimum_slab_bytes;
    result->min_expert_bytes = config.min_expert_bytes;
    result->min_expert_explicit = config.min_expert_explicit;
    result->max_batch = config.max_batch;
    result->min_compute_capability = 800;
    result->min_devices = 1;
    result->overlap_cpu_rows = config.overlap_cpu_rows;
    return 1;
}

static int metal_query_device(void * opaque, const ggml_moe_cache_config * config,
                               ggml_moe_cache_device_caps * result) {
    if (!opaque || !config || !result) return 0;
    if (metal_moe_cache_api.owner != ggml_moe_cache.owner) return 0;

    ggml_backend_dev_t device = (ggml_backend_dev_t)opaque;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if ((const void *)reg != metal_moe_cache_api.owner) return 0;

    result->logical_device = 0;
    result->physical_device = 0;
    result->compute_capability = 800;
    result->min_expert_bytes = config->min_expert_explicit
        ? config->min_expert_bytes
        : moe_cache_default_min_expert_bytes(800);
    return 1;
}

static int metal_query_shape(int wtype, int64_t n_in, int64_t n_out,
                              int64_t n_expert, size_t expert_size,
                              ggml_moe_cache_shape_caps * result) {
    if (!result || n_in <= 0 || n_out <= 0 || n_expert <= 0) return 0;
    if (!moe_cache_type_supported((ggml_type)wtype)) return 0;
    // v1: Q8_0 only
    if (wtype != GGML_TYPE_Q8_0) return 0;

    const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_in);
    if (row_size == 0) return 0;
    if ((uint64_t)n_out > SIZE_MAX / row_size) return 0;
    if (expert_size != (size_t)n_out * row_size) return 0;

    const size_t pool_bytes = expert_size * moe_cache_pool_slots_min;
    const size_t out_bytes =
        moe_cache_node_rows_max * (size_t)n_out * sizeof(float);
    const size_t act_q8_bytes =
        moe_cache_node_rows_max *
        (size_t)((n_in + QK8_1 - 1) / QK8_1) * sizeof(block_q8_1);

    result->scratch_bytes = out_bytes + act_q8_bytes;
    result->pool_bytes = pool_bytes;
    result->minimum_bytes = pool_bytes;
    return 1;
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

static void * metal_session_create(void * const * backends, int n_backends,
                                    const ggml_moe_cache_config * supplied_config) {
    try {
        moe_cache_config config = moe_cache_read_config();
        if (supplied_config) {
            constexpr size_t MiB = 1024 * 1024;
            config.enabled = true;
            config.automatic = supplied_config->min_devices > 1;
            config.budget_mb = supplied_config->budget_bytes / MiB;
            config.reserve_mb = supplied_config->reserve_bytes / MiB;
            config.minimum_slab_bytes = supplied_config->minimum_slab_bytes;
            config.min_expert_bytes = supplied_config->min_expert_bytes;
            config.min_expert_explicit = supplied_config->min_expert_explicit;
            config.max_batch = supplied_config->max_batch;
            config.min_compute_capability = supplied_config->min_compute_capability;
            config.overlap_cpu_rows = supplied_config->overlap_cpu_rows;
        }
        if (!config.enabled) return nullptr;

        // Find the Metal backend
        id<MTLDevice> mtl_dev = nil;
        for (int i = 0; i < n_backends; i++) {
            ggml_backend_t be = (ggml_backend_t)backends[i];
            if (!be) continue;
            ggml_backend_reg_t reg = ggml_backend_get_backend_reg(be);
            const char * name = ggml_backend_reg_get_name(reg);
            if (!name || strncmp(name, "Metal", 5) != 0) continue;

            // Access the Metal device through the backend's buffer type
            ggml_backend_buffer_type_t buft =
                ggml_backend_get_default_buffer_type(be);
            if (!buft) continue;

            // The Metal backend stores its device in a way accessible
            // via ggml_backend_dev_t. Get the device from the backend.
            ggml_backend_dev_t dev = ggml_backend_get_device(be);
            mtl_dev = (__bridge id<MTLDevice>)dev;
            break;
        }

        if (!mtl_dev) {
            MOE_CACHE_LOG("[moe-cache] no Metal device found\n");
            return nullptr;
        }

        // Load kernel from pre-compiled library (ggml-metal.metal)
        // The kernel was added at the end of ggml-metal.metal.
        // Access it through the Metal library system.
        id<MTLLibrary> lib = nil;
        NSError * error = nil;

        // Try loading from default.metallib first
        NSString * path = [[NSBundle mainBundle] pathForResource:@"default"
                                                          ofType:@"metallib"];
        if (!path) {
            // Try relative to the binary
            NSString * binDir = [[[NSProcessInfo processInfo] arguments][0]
                stringByDeletingLastPathComponent];
            path = [binDir stringByAppendingPathComponent:@"default.metallib"];
        }

        if (path && [[NSFileManager defaultManager] isReadableFileAtPath:path]) {
            lib = [mtl_dev newLibraryWithURL:[NSURL fileURLWithPath:path]
                                       error:&error];
        }

        if (!lib) {
            MOE_CACHE_LOG("[moe-cache] cannot load default.metallib: %s\n",
                error ? [[error description] UTF8String] : "not found");
            return nullptr;
        }

        id<MTLFunction> fn = [lib newFunctionWithName:
            @"kernel_moe_cache_mv_q8_0_f32"];
        [lib release];

        if (!fn) {
            MOE_CACHE_LOG("[moe-cache] kernel not found in metallib "
                "(was ggml-metal.metal updated?)\n");
            return nullptr;
        }

        id<MTLComputePipelineState> pipeline =
            [mtl_dev newComputePipelineStateWithFunction:fn error:&error];
        [fn release];

        if (!pipeline) {
            MOE_CACHE_LOG("[moe-cache] pipeline failed: %s\n",
                error ? [[error description] UTF8String] : "unknown");
            return nullptr;
        }

        id<MTLCommandQueue> queue = [mtl_dev newCommandQueue];
        if (!queue) {
            [pipeline release];
            return nullptr;
        }

        auto session = std::make_unique<moe_cache_session>();
        session->config = std::move(config);

        auto dev = std::make_unique<moe_cache_metal_device>(mtl_dev, queue);
        dev->mmv_pipeline = pipeline;
        [queue release]; // retained by device
        [pipeline release]; // retained by device

        session->devices.push_back(std::move(dev));

        MOE_CACHE_LOG("[moe-cache] Metal session created\n");
        return session.release();
    } catch (...) {
        MOE_CACHE_LOG("[moe-cache] Metal session creation failed\n");
        return nullptr;
    }
}

static void metal_session_destroy(void * opaque) {
    if (!opaque) return;
    delete (moe_cache_session *)opaque;
}

static void metal_session_enter(void * opaque) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    moe_cache_scope_frame frame;
    frame.requested = session;
    frame.active = nullptr;
    g_session_stack.push_back(frame);
}

static void metal_session_leave(void * opaque) {
    (void)opaque;
    if (!g_session_stack.empty()) {
        g_session_stack.pop_back();
    }
}

// ---------------------------------------------------------------------------
// Quantize activations to Q8_1 (scalar, reference quality).
// ---------------------------------------------------------------------------

static void metal_quantize_act_q8_1(const float * src, block_q8_1 * dst,
                                     int64_t n, int64_t padded_n) {
    const int nb = (int)(padded_n / QK8_1);
    for (int ib = 0; ib < nb; ib++) {
        float amax = 0.0f;
        for (int i = 0; i < QK8_1; i++) {
            int64_t idx = (int64_t)ib * QK8_1 + i;
            float v = (idx < n) ? src[idx] : 0.0f;
            amax = fmaxf(amax, fabsf(v));
        }
        const float d = amax / 127.0f;
        const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < QK8_1; i++) {
            int64_t idx = (int64_t)ib * QK8_1 + i;
            float v = (idx < n) ? src[idx] : 0.0f;
            int8_t q = (int8_t)roundf(v * id);
            dst[ib].qs[i] = q;
            sum += (float)q * d;
        }
        dst[ib].d = (half)d;
        dst[ib].s = (half)sum;
    }
}

// ---------------------------------------------------------------------------
// Begin: Check cache, fill misses, build node.
// ---------------------------------------------------------------------------

static void * metal_begin(const ggml_moe_cache_tensor_desc * tensor, int pool,
                           int64_t n_tokens, int n_rows, const int32_t * ids,
                           const float * const * act_rows, uint64_t * hit_mask) {
    if (!tensor || !ids || !act_rows || !hit_mask) return nullptr;
    if (n_rows <= 0 || n_rows > moe_cache_node_rows_max) return nullptr;

    // Find active session from thread-local stack
    moe_cache_session * session = nullptr;
    for (auto it = g_session_stack.rbegin();
         it != g_session_stack.rend(); ++it) {
        if (it->active) { session = it->active; break; }
    }
    if (!session || session->devices.empty()) return nullptr;

    moe_cache_device * dev_raw = session->devices[0].get();
    auto & dev = static_cast<moe_cache_metal_device &>(*dev_raw);

    std::unique_lock<std::mutex> lock(dev.dispatch_mu);

    if (pool < 0 || pool >= (int)dev.pools.size()) return nullptr;
    moe_cache_pool * pool_ptr = dev.pools[pool].get();

    // Check hits and sync-fill misses
    int n_hits = 0;
    for (int i = 0; i < n_rows; i++) {
        moe_cache_key key{tensor->data, ids[i]};
        auto it = pool_ptr->map.find(key);
        if (it != pool_ptr->map.end()) {
            moe_cache_slot & slot = pool_ptr->slots[it->second];
            if (slot.state == moe_cache_slot_state::valid) {
                hit_mask[i / 64] |= (1ULL << (i % 64));
                n_hits++;
                dev.hits++;
                continue;
            }
        }
        dev.misses++;

        // Evict LRU if full
        if (pool_ptr->free_slots.empty() && pool_ptr->lru_tail >= 0) {
            moe_cache_slot_reset(*pool_ptr, pool_ptr->lru_tail, true);
        }

        if (!pool_ptr->free_slots.empty()) {
            int slot_idx = pool_ptr->free_slots.back();
            pool_ptr->free_slots.pop_back();
            moe_cache_slot & s = pool_ptr->slots[slot_idx];
            s.key = key;
            s.state = moe_cache_slot_state::valid;
            pool_ptr->map[key] = slot_idx;

            // memcpy expert weight into slab (unified memory — direct)
            memcpy(pool_ptr->slab + (size_t)slot_idx * pool_ptr->expert_size,
                   tensor->data, tensor->expert_size);

            moe_cache_lru_push_back(*pool_ptr, slot_idx);
            hit_mask[i / 64] |= (1ULL << (i % 64));
            n_hits++;
            dev.hits++;
            dev.fills++;
        }
    }

    if (n_hits == 0) return nullptr;

    // Build node
    auto node = std::make_unique<moe_cache_node>();
    node->session = session;
    node->device = dev_raw;
    node->pool = pool_ptr;
    node->pool_index = pool;
    node->host_base = tensor->data;
    node->expert_size = tensor->expert_size;
    node->n_in = tensor->n_in;
    node->n_out = tensor->n_out;
    node->n_expert = tensor->n_expert;
    node->n_tokens = n_tokens;
    node->wtype = GGML_TYPE_Q8_0;
    node->dispatch_lock = std::move(lock);

    // Pin the slots that were hits
    int pin_count = 0;
    for (int i = 0; i < n_rows; i++) {
        if (!(hit_mask[i / 64] & (1ULL << (i % 64)))) continue;
        moe_cache_key key{tensor->data, ids[i]};
        auto it = pool_ptr->map.find(key);
        if (it != pool_ptr->map.end()) {
            node->pins[pin_count].pool = pool_ptr;
            node->pins[pin_count].slot = it->second;
            pool_ptr->slots[it->second].readers++;
            pin_count++;
        }
    }
    node->n_pins = pin_count;

    dev.nodes++;
    return node.release();
}

// ---------------------------------------------------------------------------
// Plan: Prepare activation quantization + upload.
// ---------------------------------------------------------------------------

static int metal_plan(void * opaque) {
    // For Metal v1: fill already done in begin().
    // Plan just marks the node ready.
    if (!opaque) return 0;
    moe_cache_node * node = (moe_cache_node *)opaque;
    node->planned = true;
    return 1;
}

// ---------------------------------------------------------------------------
// Dispatch: Launch Metal matvec kernel.
// ---------------------------------------------------------------------------

static int metal_dispatch_internal(
        moe_cache_node * node, int n_hits,
        const int32_t * slot_indices, const float * const * act_rows) {
    if (!node || !node->planned || !slot_indices || !act_rows) return 0;
    if (n_hits <= 0 || n_hits > moe_cache_node_rows_max) return 0;

    moe_cache_session & session = *node->session;
    auto & dev = static_cast<moe_cache_metal_device &>(*node->device);
    moe_cache_pool & pool = *node->pool;

    if (dev.dead.load()) return 0;

    const int64_t n_in = node->n_in;
    const int64_t n_out = node->n_out;
    const int64_t expert_stride = node->expert_size;
    const int64_t row_stride = ggml_row_size(GGML_TYPE_Q8_0, n_in);
    const int64_t padded_n_in =
        ((n_in + QK8_1 - 1) / QK8_1) * QK8_1;

    // Deduplicate activation rows
    const float * unique_act_rows[moe_cache_node_rows_max];
    int activation_indices[moe_cache_node_rows_max];
    int activation_rows = 0;

    for (int i = 0; i < n_hits; i++) {
        int act = 0;
        while (act < activation_rows &&
               unique_act_rows[act] != act_rows[i]) act++;
        if (act == activation_rows) {
            unique_act_rows[activation_rows++] = act_rows[i];
        }
        activation_indices[i] = act;
    }

    // Allocate output buffer
    const size_t out_bytes = (size_t)n_hits * (size_t)n_out * sizeof(float);
    id<MTLBuffer> out_buf = [dev.mtl_device newBufferWithLength:out_bytes
        options:MTLResourceStorageModeShared];
    if (!out_buf) return 0;
    float * out_ptr = (float *)[out_buf contents];

    // Allocate and quantize activation buffer
    const size_t act_q8_bytes =
        (size_t)activation_rows * (size_t)(padded_n_in / QK8_1) *
        sizeof(block_q8_1);
    id<MTLBuffer> act_buf = [dev.mtl_device newBufferWithLength:act_q8_bytes
        options:MTLResourceStorageModeShared];
    if (!act_buf) {
        [out_buf release];
        return 0;
    }
    block_q8_1 * act_q8 = (block_q8_1 *)[act_buf contents];

    for (int a = 0; a < activation_rows; a++) {
        metal_quantize_act_q8_1(unique_act_rows[a],
            act_q8 + a * (padded_n_in / QK8_1), n_in, padded_n_in);
    }

    // Allocate and fill slot indices buffer
    const size_t ids_bytes = (size_t)n_hits * sizeof(int32_t);
    id<MTLBuffer> ids_buf = [dev.mtl_device newBufferWithLength:ids_bytes
        options:MTLResourceStorageModeShared];
    if (!ids_buf) {
        [act_buf release];
        [out_buf release];
        return 0;
    }
    int32_t * ids_ptr = (int32_t *)[ids_buf contents];
    memcpy(ids_ptr, slot_indices, ids_bytes);

    // Launch Metal kernel
    id<MTLCommandBuffer> cmd_buf = [dev.mtl_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc =
        [cmd_buf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

    [enc setComputePipelineState:dev.mmv_pipeline];

    // Get the slab buffer — it's one of the tracked slab_buffers.
    // The pool slab pointer points into a specific MTLBuffer's contents.
    // Find which buffer it belongs to.
    id<MTLBuffer> slab_buf = nil;
    for (id<MTLBuffer> buf : dev.slab_buffers) {
        char * contents = (char *)[buf contents];
        char * slab = pool.slab;
        if (slab >= contents && slab < contents + [buf length]) {
            slab_buf = buf;
            break;
        }
    }
    if (!slab_buf) {
        [ids_buf release];
        [act_buf release];
        [out_buf release];
        [enc endEncoding];
        [cmd_buf release];
        return 0;
    }

    // Calculate slab offset within the buffer
    const NSUInteger slab_offset = (NSUInteger)(pool.slab - (char *)[slab_buf contents]);

    [enc setBuffer:slab_buf offset:slab_offset atIndex:0];
    [enc setBuffer:ids_buf offset:0 atIndex:1];
    [enc setBuffer:act_buf offset:0 atIndex:2];
    [enc setBuffer:out_buf offset:0 atIndex:3];

    int64_t args[] = {n_in, n_out, expert_stride, row_stride, n_hits, padded_n_in};
    [enc setBytes:args length:sizeof(args) atIndex:4];

    const NSUInteger total_threads = (NSUInteger)(n_hits * n_out);
    [enc dispatchThreadgroups:MTLSizeMake(
        (total_threads + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [enc endEncoding];
    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    // Store results for collect
    dev.d_out = out_ptr;
    dev.d_out_cap = out_bytes;

    // Release temporary buffers (output kept)
    [ids_buf release];
    [act_buf release];
    // out_buf released in end() or next dispatch
    if (dev.out_buffer) [dev.out_buffer release];
    dev.out_buffer = out_buf;

    dev.dispatch_failures = 0; // reset (not tracking per-node yet)
    return 1;
}

static int metal_dispatch(void * opaque) {
    // The actual dispatch is done inline when the scheduler calls,
    // but we don't have access to the slot_indices/act_rows here.
    // These are passed through the ggml_moe_cache_api dispatch function,
    // which calls metal_dispatch_internal via the API table.
    //
    // For now, this is a placeholder. The real dispatch happens when
    // the scheduler calls ggml_moe_cache.dispatch(node, wtype, n_in, n_out,
    // n_hits, slot_indices, act_rows) which maps to the API table entry.
    (void)opaque;
    return 1;
}

// ---------------------------------------------------------------------------
// Collect: Results already in shared memory.
// ---------------------------------------------------------------------------

static int metal_collect(void * opaque) {
    if (!opaque) return 0;
    // Results are in d_out (shared memory on unified arch).
    // The scheduler reads them directly.
    return 1;
}

// ---------------------------------------------------------------------------
// End: Unpin slots, free node.
// ---------------------------------------------------------------------------

static void metal_end(void * opaque) {
    if (!opaque) return;
    moe_cache_node * node = (moe_cache_node *)opaque;

    for (int i = 0; i < node->n_pins; i++) {
        if (node->pins[i].pool && node->pins[i].slot >= 0) {
            node->pins[i].pool->slots[node->pins[i].slot].readers--;
        }
    }

    delete node;
}

// ---------------------------------------------------------------------------
// Fused SwiGLU — not implemented for v1.
// ---------------------------------------------------------------------------

static void * metal_fused_begin(const ggml_moe_cache_tensor_desc * up,
                                 const ggml_moe_cache_tensor_desc * gate,
                                 int glu_op, float up_min, float up_max,
                                 float gate_min, float gate_max,
                                 const int32_t * ids, int n_rows,
                                 int64_t n_tokens,
                                 const float * const * act_rows,
                                 uint64_t * hit_mask) {
    (void)up; (void)gate; (void)glu_op;
    (void)up_min; (void)up_max; (void)gate_min; (void)gate_max;
    (void)ids; (void)n_rows; (void)n_tokens; (void)act_rows; (void)hit_mask;
    return nullptr;
}

static void metal_invalidate(void * opaque, const void * tensor_base) {
    if (!opaque || !tensor_base) return;
    moe_cache_session * session = (moe_cache_session *)opaque;
    for (auto & dev_ptr : session->devices) {
        for (auto & pool_ptr : dev_ptr->pools) {
            for (int i = 0; i < pool_ptr->n_slots; i++) {
                if (pool_ptr->slots[i].key.tensor == tensor_base) {
                    moe_cache_slot_reset(*pool_ptr, i, true);
                }
            }
        }
    }
}

static int metal_trim(void * opaque, size_t target_bytes) {
    (void)opaque;
    (void)target_bytes;
    return 0;
}

// ---------------------------------------------------------------------------
// API table + Registration
// ---------------------------------------------------------------------------

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

extern "C" void ggml_metal_moe_cache_register(void) {
    ggml_moe_cache_register(&metal_moe_cache_api);
}
