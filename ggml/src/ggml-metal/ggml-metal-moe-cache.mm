// Metal MoE Expert Cache.
//
// Keeps the hottest CPU-resident MoE expert weights resident in Metal
// shared (unified) memory, so decode-time expert matvecs run on the GPU
// instead of the CPU path with PCIe-round-trip weight reads.
//
// This implements the real ggml_moe_cache_api contract (see
// ggml-backend-moe-cache.h). The scheduler owns one cache session; the
// CPU MUL_MAT_ID path calls begin/plan/dispatch/collect/end per node.
//
// v1 notes:
//   - synchronous fills (no worker thread): plan() copies missed experts
//     into the slab before returning, so a miss still serves this node
//   - one pool per (expert_size, wtype), allocated lazily in begin()
//   - unified memory: fills are plain memcpy, results live in shared
//     buffers readable by the CPU
//   - fused SwiGLU returns NULL (stock CPU path handles the node)
//
// Register by calling ggml_metal_moe_cache_register() from
// ggml_backend_metal_reg() after the backend reg struct is set up.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ggml-metal.h"
#include "ggml-metal-context.h"
#include "ggml-metal-device.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "../ggml-quants.h"
#include "../ggml-moe-cache-common.h"

// Thread-local session stack (owned by this backend; independent of CUDA's).
static thread_local std::vector<moe_cache_scope_frame> g_session_stack;
static thread_local int g_session_suppressed = 0;

// Global session registry for invalidate()/teardown paths.
static std::mutex g_registry_mu;
static std::unordered_set<moe_cache_session *> g_sessions;
// Live-session count, so invalidate() can bail before taking g_registry_mu.
// invalidate runs on every backend buffer write; with no cache active that was
// one mutex acquire per write during model load.
static std::atomic<size_t> g_session_count{0};

// Backend registration object this provider was registered under.
static const void * g_moe_cache_owner = nullptr;

// ---------------------------------------------------------------------------
// Metal device extension
// ---------------------------------------------------------------------------

struct moe_cache_metal_device : public moe_cache_device {
    moe_cache_metal_device(id<MTLDevice> dev, ggml_metal_device_t ctx)
        : moe_cache_device(0, 0), mtl_device(dev), ctx_dev(ctx) {
        [mtl_device retain];
    }

    ~moe_cache_metal_device() {
        free_slabs();
        if (out_buffer) {
            [out_buffer release];
            out_buffer = nil;
        }
        if (mtl_queue) {
            [mtl_queue release];
            mtl_queue = nil;
        }
        [mtl_device release];
    }

    id<MTLDevice> mtl_device;
    // Backend device context; the library is compiled from it on first use.
    ggml_metal_device_t ctx_dev = nullptr;
    // Command queue, library, and pipelines are created lazily on the first
    // begin, so a context that never uses the cache allocates no GPU objects.
    id<MTLCommandQueue> mtl_queue = nil;
    std::once_flag init_once;
    bool init_ok = false;
    ggml_metal_library_t lib = nullptr;
    struct ggml_metal_pipeline_with_params mmv_pipeline_q8_0;
    struct ggml_metal_pipeline_with_params mmv_pipeline_q4_0;
    struct ggml_metal_pipeline_with_params mmv_pipeline_q4_K;
    struct ggml_metal_pipeline_with_params mmv_pipeline_q6_K;
    struct ggml_metal_pipeline_with_params mmv_pipeline_q5_K;
    struct ggml_metal_pipeline_with_params mmv_pipeline_q1_0;
    struct ggml_metal_pipeline_with_params mmv_pipeline_q2_0;
    struct ggml_metal_pipeline_with_params mmv_pipeline_q4_1;
    struct ggml_metal_pipeline_with_params mmv_pipeline_q5_0;
    struct ggml_metal_pipeline_with_params mmv_pipeline_q5_1;
    struct ggml_metal_pipeline_with_params mmv_pipeline_q2_K;
    struct ggml_metal_pipeline_with_params mmv_pipeline_q3_K;
    struct ggml_metal_pipeline_with_params mmv_pipeline_iq2_xxs;
    struct ggml_metal_pipeline_with_params mmv_pipeline_iq2_xs;
    struct ggml_metal_pipeline_with_params mmv_pipeline_iq2_s;
    struct ggml_metal_pipeline_with_params mmv_pipeline_iq3_xxs;
    struct ggml_metal_pipeline_with_params mmv_pipeline_iq3_s;
    struct ggml_metal_pipeline_with_params mmv_pipeline_iq1_s;
    struct ggml_metal_pipeline_with_params mmv_pipeline_iq1_m;
    struct ggml_metal_pipeline_with_params mmv_pipeline_iq4_nl;
    struct ggml_metal_pipeline_with_params mmv_pipeline_iq4_xs;
    struct ggml_metal_pipeline_with_params mmv_pipeline_mxfp4;
    struct ggml_metal_pipeline_with_params mmv_pipeline_nvfp4;

    // Tracked MTLBuffers for slab pools.
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
};

// ---------------------------------------------------------------------------
// Slab allocation — MTLBuffer with shared storage (unified memory).
// ---------------------------------------------------------------------------

static char * metal_slab_alloc(moe_cache_metal_device & dev, size_t bytes) {
    id<MTLBuffer> buf = [dev.mtl_device newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];
    if (!buf) {
        return nullptr;
    }
    dev.slab_buffers.push_back(buf);
    return (char *)[buf contents];
}

// ---------------------------------------------------------------------------
// Query functions
// ---------------------------------------------------------------------------

static int metal_query_config(int automatic, size_t budget_mib,
                               ggml_moe_cache_config * result) {
    if (!result) {
        return 0;
    }

    moe_cache_config config = moe_cache_read_config();
    if (automatic >= 0) {
        config.enabled = true;
        config.automatic = automatic != 0;
        moe_cache_apply_mode_defaults(config);
        config.min_compute_capability =
            moe_cache_min_compute_capability(config.automatic);
    }
    if (budget_mib > 0) {
        config.budget_mb = budget_mib;
    }
    if (!config.enabled || config.budget_mb > (SIZE_MAX >> 20) ||
        config.reserve_mb > (SIZE_MAX >> 20)) {
        return 0;
    }

    result->budget_bytes = config.budget_mb << 20;
    result->reserve_bytes = config.reserve_mb << 20;
    result->minimum_slab_bytes = config.minimum_slab_bytes;
    result->min_expert_bytes = config.min_expert_bytes;
    result->min_expert_explicit = config.min_expert_explicit;
    result->max_batch = config.max_batch;
    result->min_compute_capability = config.min_compute_capability;
    result->min_devices = 1; // Metal always has exactly one device
    result->overlap_cpu_rows = config.overlap_cpu_rows;
    return 1;
}

static int metal_query_device(void * opaque, const ggml_moe_cache_config * config,
                               ggml_moe_cache_device_caps * result) {
    if (!opaque || !config || !result || !g_moe_cache_owner) {
        return 0;
    }

    ggml_backend_dev_t device = (ggml_backend_dev_t)opaque;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if ((const void *)reg != g_moe_cache_owner) {
        return 0;
    }

    result->logical_device = 0;
    result->physical_device = 0;
    result->compute_capability = 800; // Apple Silicon ~ Ampere-class
    result->min_expert_bytes = config->min_expert_explicit
        ? config->min_expert_bytes
        : moe_cache_default_min_expert_bytes(800);
    return 1;
}

static int metal_query_shape(int wtype, int64_t n_in, int64_t n_out,
                              int64_t n_expert, size_t expert_size,
                              ggml_moe_cache_shape_caps * result) {
    if (!result || n_in <= 0 || n_out <= 0 || n_expert <= 0) {
        return 0;
    }
    // canonical list from ggml-backend-moe-cache.h; single source of truth
    if (!ggml_moe_cache_wtype_supported(wtype)) {
        return 0;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size != (size_t)n_out * row_size ||
        expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return 0;
    }

    const size_t out_bytes =
        moe_cache_node_rows_max * (size_t)n_out * sizeof(float);
    const size_t act_q8_bytes =
        moe_cache_node_rows_max *
        (size_t)((n_in + QK8_1 - 1) / QK8_1) * sizeof(block_q8_1);
    const size_t scratch_bytes = out_bytes + act_q8_bytes;
    const size_t pool_bytes = expert_size * moe_cache_pool_slots_min;
    if (pool_bytes > SIZE_MAX - scratch_bytes) {
        return 0;
    }
    result->scratch_bytes = scratch_bytes;
    result->pool_bytes = pool_bytes;
    result->minimum_bytes = scratch_bytes + pool_bytes;
    return 1;
}

// ---------------------------------------------------------------------------
// Pool lifecycle
// ---------------------------------------------------------------------------

static moe_cache_pool * metal_find_or_create_pool(
        moe_cache_metal_device & dev, moe_cache_session & session,
        size_t expert_size, int wtype, int64_t n_expert, size_t budget_bytes) {
    const int existing = moe_cache_find_pool(dev, expert_size, wtype);
    if (existing >= 0) {
        return dev.pools[existing].get();
    }

    if (moe_cache_fail(session, "slab")) {
        MOE_CACHE_LOG("[moe-cache] Metal: skipped %zu KiB expert pool: allocation failed\n",
                expert_size >> 10);
        return nullptr;
    }

    size_t slots = budget_bytes / expert_size;
    if (slots < moe_cache_pool_slots_min) {
        return nullptr;
    }
    if ((uint64_t)n_expert > 0 && slots > (size_t)n_expert) {
        slots = (size_t)n_expert;
    }
    if (slots < moe_cache_pool_slots_min) {
        return nullptr;
    }
    if (slots > (size_t)INT_MAX) {
        slots = INT_MAX;
    }
    const size_t slab_bytes = slots * expert_size;

    char * slab = metal_slab_alloc(dev, slab_bytes);
    if (!slab) {
        MOE_CACHE_LOG("[moe-cache] Metal: failed to allocate %zu MiB expert pool\n",
                slab_bytes >> 20);
        return nullptr;
    }

    try {
        std::unique_ptr<moe_cache_pool> pool(new moe_cache_pool());
        pool->expert_size = expert_size;
        pool->wtype = wtype;
        pool->slab = slab;
        pool->n_slots = (int)slots;
        pool->covers_all_entries = (uint64_t)slots >= (uint64_t)n_expert;
        pool->slots.resize(slots);
        pool->free_slots.reserve(slots);
        pool->map.reserve(slots);
        for (int index = (int)slots - 1; index >= 0; index--) {
            pool->free_slots.push_back(index);
        }
        dev.allocated_bytes += slab_bytes;
        dev.pools.push_back(std::move(pool));
        MOE_CACHE_LOG("[moe-cache] Metal%d pool[%d]: type=%s expert=%zu KiB slots=%zu entries=%lld coverage=%s total=%zu MiB\n",
                dev.physical, (int)dev.pools.size() - 1,
                ggml_type_name((ggml_type)wtype), expert_size >> 10,
                slots, (long long)n_expert,
                dev.pools.back()->covers_all_entries ? "complete" : "partial",
                slab_bytes >> 20);
        bool expected = false;
        if (session.enabled_announced.compare_exchange_strong(expected, true)) {
            MOE_CACHE_LOG("[moe-cache] enabled: first pool allocated on Metal%d\n",
                    dev.physical);
        }
        return dev.pools.back().get();
    } catch (...) {
        // remove the tracked buffer and restore the pool list
        if (!dev.slab_buffers.empty()) {
            id<MTLBuffer> buf = dev.slab_buffers.back();
            [buf release];
            dev.slab_buffers.pop_back();
        }
        dev.allocated_bytes -= slab_bytes;
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

// Get-or-compile a moe-cache kernel pipeline, logging the name on failure.
// The backend caches pipelines lazily; get_pipeline alone only returns
// already-compiled ones, so every lookup on a fresh library would miss.
static struct ggml_metal_pipeline_with_params metal_pipeline_get(
        ggml_metal_library_t lib, const char * name) {
    struct ggml_metal_pipeline_with_params res =
        ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    if (!res.pipeline) {
        MOE_CACHE_LOG("[moe-cache] Metal: moe cache kernel missing: %s\n", name);
    }
    return res;
}

// Compile the moe-cache pipelines and create the command queue on first use.
// session_create only keeps bookkeeping state, so a context that never uses
// the cache pays no GPU setup cost.
static bool metal_device_ensure_ready(moe_cache_metal_device & dev, size_t budget_mb) {
    std::call_once(dev.init_once, [&]() {
        // Load the shared kernel library (same embedded source as the
        // backend; contains kernel_moe_cache_mv_* from ggml-metal.metal).
        ggml_metal_library_t lib = ggml_metal_library_init(dev.ctx_dev);
        if (!lib) {
            MOE_CACHE_LOG("[moe-cache] Metal library init failed\n");
            dev.dead.store(true);
            return;
        }

        struct ggml_metal_pipeline_with_params p_q8_0 =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_q8_0_f32");
        struct ggml_metal_pipeline_with_params p_q4_0 =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_q4_0_f32");
        struct ggml_metal_pipeline_with_params p_q4_K =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_q4_K_f32");
        struct ggml_metal_pipeline_with_params p_q6_K =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_q6_K_f32");
        struct ggml_metal_pipeline_with_params p_q5_K =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_q5_K_f32");
        struct ggml_metal_pipeline_with_params p_q1_0 =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_q1_0_f32");
        struct ggml_metal_pipeline_with_params p_q2_0 =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_q2_0_f32");
        struct ggml_metal_pipeline_with_params p_q4_1 =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_q4_1_f32");
        struct ggml_metal_pipeline_with_params p_q5_0 =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_q5_0_f32");
        struct ggml_metal_pipeline_with_params p_q5_1 =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_q5_1_f32");
        struct ggml_metal_pipeline_with_params p_q2_K =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_q2_K_f32");
        struct ggml_metal_pipeline_with_params p_q3_K =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_q3_K_f32");
        struct ggml_metal_pipeline_with_params p_iq2_xxs =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_iq2_xxs_f32");
        struct ggml_metal_pipeline_with_params p_iq2_xs =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_iq2_xs_f32");
        struct ggml_metal_pipeline_with_params p_iq2_s =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_iq2_s_f32");
        struct ggml_metal_pipeline_with_params p_iq3_xxs =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_iq3_xxs_f32");
        struct ggml_metal_pipeline_with_params p_iq3_s =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_iq3_s_f32");
        struct ggml_metal_pipeline_with_params p_iq1_s =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_iq1_s_f32");
        struct ggml_metal_pipeline_with_params p_iq1_m =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_iq1_m_f32");
        struct ggml_metal_pipeline_with_params p_iq4_nl =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_iq4_nl_f32");
        struct ggml_metal_pipeline_with_params p_iq4_xs =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_iq4_xs_f32");
        struct ggml_metal_pipeline_with_params p_mxfp4 =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_mxfp4_f32");
        struct ggml_metal_pipeline_with_params p_nvfp4 =
            metal_pipeline_get(lib, "kernel_moe_cache_mv_nvfp4_f32");
        if (!p_q8_0.pipeline || !p_q4_0.pipeline ||
            !p_q4_K.pipeline || !p_q6_K.pipeline || !p_q5_K.pipeline ||
            !p_q1_0.pipeline || !p_q2_0.pipeline || !p_q4_1.pipeline ||
            !p_q5_0.pipeline || !p_q5_1.pipeline || !p_q2_K.pipeline ||
            !p_q3_K.pipeline || !p_iq2_xxs.pipeline || !p_iq2_xs.pipeline ||
            !p_iq2_s.pipeline || !p_iq3_xxs.pipeline || !p_iq3_s.pipeline ||
            !p_iq1_s.pipeline || !p_iq1_m.pipeline || !p_iq4_nl.pipeline ||
            !p_iq4_xs.pipeline || !p_mxfp4.pipeline || !p_nvfp4.pipeline) {
            MOE_CACHE_LOG("[moe-cache] Metal: one or more moe cache kernels missing\n");
            ggml_metal_library_free(lib);
            dev.dead.store(true);
            return;
        }

        id<MTLCommandQueue> queue = [dev.mtl_device newCommandQueue];
        if (!queue) {
            ggml_metal_library_free(lib);
            dev.dead.store(true);
            return;
        }

        dev.mtl_queue = queue;
        dev.lib = lib;
        dev.mmv_pipeline_q8_0 = p_q8_0;
        dev.mmv_pipeline_q4_0 = p_q4_0;
        dev.mmv_pipeline_q4_K = p_q4_K;
        dev.mmv_pipeline_q6_K = p_q6_K;
        dev.mmv_pipeline_q5_K = p_q5_K;
        dev.mmv_pipeline_q1_0 = p_q1_0;
        dev.mmv_pipeline_q2_0 = p_q2_0;
        dev.mmv_pipeline_q4_1 = p_q4_1;
        dev.mmv_pipeline_q5_0 = p_q5_0;
        dev.mmv_pipeline_q5_1 = p_q5_1;
        dev.mmv_pipeline_q2_K = p_q2_K;
        dev.mmv_pipeline_q3_K = p_q3_K;
        dev.mmv_pipeline_iq2_xxs = p_iq2_xxs;
        dev.mmv_pipeline_iq2_xs = p_iq2_xs;
        dev.mmv_pipeline_iq2_s = p_iq2_s;
        dev.mmv_pipeline_iq3_xxs = p_iq3_xxs;
        dev.mmv_pipeline_iq3_s = p_iq3_s;
        dev.mmv_pipeline_iq1_s = p_iq1_s;
        dev.mmv_pipeline_iq1_m = p_iq1_m;
        dev.mmv_pipeline_iq4_nl = p_iq4_nl;
        dev.mmv_pipeline_iq4_xs = p_iq4_xs;
        dev.mmv_pipeline_mxfp4 = p_mxfp4;
        dev.mmv_pipeline_nvfp4 = p_nvfp4;
        dev.init_ok = true;
        MOE_CACHE_LOG("[moe-cache] Metal session ready (budget=%zu MiB)\n", budget_mb);
    });
    return dev.init_ok;
}

static void * metal_session_create(void * const * backends, int n_backends,
                                    const ggml_moe_cache_config * supplied_config) {
    try {
        moe_cache_config config = moe_cache_read_config();
        if (supplied_config) {
            constexpr size_t MiB = 1024 * 1024;
            if (supplied_config->budget_bytes % MiB != 0 ||
                supplied_config->reserve_bytes % MiB != 0 ||
                supplied_config->minimum_slab_bytes % MiB != 0 ||
                supplied_config->min_expert_bytes == 0 ||
                supplied_config->min_expert_explicit < 0 ||
                supplied_config->min_expert_explicit > 1 ||
                supplied_config->max_batch < 1 ||
                supplied_config->max_batch > moe_cache_batch_max ||
                supplied_config->min_devices < 1 ||
                supplied_config->min_compute_capability < 0 ||
                supplied_config->min_compute_capability > 999 ||
                supplied_config->overlap_cpu_rows < -1 ||
                supplied_config->overlap_cpu_rows > 8) {
                return nullptr;
            }
            config.enabled = true;
            config.automatic = supplied_config->minimum_slab_bytes > 0;
            config.budget_mb = supplied_config->budget_bytes / MiB;
            config.reserve_mb = supplied_config->reserve_bytes / MiB;
            config.minimum_slab_bytes = supplied_config->minimum_slab_bytes;
            config.min_expert_bytes = supplied_config->min_expert_bytes;
            config.min_expert_explicit = supplied_config->min_expert_explicit;
            config.max_batch = supplied_config->max_batch;
            config.min_compute_capability = supplied_config->min_compute_capability;
            config.overlap_cpu_rows = supplied_config->overlap_cpu_rows;
        }
        if (!config.enabled) {
            return nullptr;
        }
        // A zero budget can never create a pool; bail before retaining the
        // MTLDevice, which would trigger an AppleParavirtCommandQueue retain
        // cycle reported by `leaks` on macOS CI.
        if (!supplied_config && config.budget_mb == 0) {
            return nullptr;
        }

        // Find the Metal backend among the scheduler's backends.
        ggml_metal_device_t ctx_dev = nullptr;
        id<MTLDevice> mtl_dev = nil;
        for (int i = 0; i < n_backends; i++) {
            ggml_backend_t be = (ggml_backend_t)backends[i];
            if (!be || !ggml_backend_is_metal(be)) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(be);
            if (!dev || !dev->context) {
                continue;
            }
            ctx_dev = (ggml_metal_device_t)dev->context;
            mtl_dev = (__bridge id<MTLDevice>)ggml_metal_device_get_obj(ctx_dev);
            break;
        }
        if (!mtl_dev) {
            MOE_CACHE_LOG("[moe-cache] no Metal device found\n");
            return nullptr;
        }

        // GPU resources (command queue, library, pipelines) are created
        // lazily on the first begin; a context that never uses the cache
        // allocates none of them.
        std::unique_ptr<moe_cache_session> session(new (std::nothrow) moe_cache_session());
        if (!session) {
            return nullptr;
        }
        session->config = std::move(config);

        std::unique_ptr<moe_cache_metal_device> dev(new (std::nothrow)
                moe_cache_metal_device(mtl_dev, ctx_dev));
        if (!dev) {
            return nullptr;
        }
        session->devices.push_back(std::move(dev));

        moe_cache_session * result = session.get();
        try {
            std::lock_guard<std::mutex> lock(g_registry_mu);
            g_sessions.insert(result);
            g_session_count.store(g_sessions.size(), std::memory_order_release);
        } catch (...) {
            return nullptr;
        }
        session.release();
        return result;
    } catch (...) {
        MOE_CACHE_LOG("[moe-cache] Metal session creation failed\n");
        return nullptr;
    }
}

// Teardown statistics, same field names as CUDA so the log contract is
// backend-independent. Only logged when the session did any cache work.
static void metal_log_stats(moe_cache_metal_device & dev) {
    size_t used = 0;
    size_t slots = 0;
    for (const auto & pool_ptr : dev.pools) {
        const moe_cache_pool & pool = *pool_ptr;
        slots += pool.n_slots;
        used += pool.n_slots - pool.free_slots.size();
    }
    const long long total = dev.hits + dev.misses;
    MOE_CACHE_LOG("[moe-cache] Metal%d hits=%lld/%lld (%.1f%%) used=%zu/%zu enqueued=%lld filled=%lld fill-fail=%lld evictions=%lld skips=%lld admission=%lld dispatch-fail=%lld collect-fail=%lld bypass=%lld\n",
            dev.physical, dev.hits, total,
            total ? 100.0 * (double)dev.hits / (double)total : 0.0,
            used, slots, dev.inserts, dev.fills, dev.fill_failures,
            dev.evictions, dev.insert_skips, dev.admission_skips,
            dev.dispatch_failures, dev.collect_failures, dev.contention_bypasses);
}

static void metal_session_destroy(void * opaque) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
    }
    {
        // Stop new scopes/nodes and wait for in-flight work to drain before
        // touching device resources a CPU worker may still be using.
        std::unique_lock<std::mutex> lock(session->mu);
        session->stopping = true;
        session->cv.notify_all();
        session->idle_cv.wait(lock, [&] {
            return session->active_scopes == 0 && session->active_nodes == 0;
        });
    }
    {
        std::lock_guard<std::mutex> lock(g_registry_mu);
        g_sessions.erase(session);
        g_session_count.store(g_sessions.size(), std::memory_order_release);
    }
    for (auto & dev_ptr : session->devices) {
        moe_cache_metal_device & dev =
            static_cast<moe_cache_metal_device &>(*dev_ptr);
        if (dev.nodes > 0 || dev.dispatch_failures > 0 ||
            dev.collect_failures > 0) {
            metal_log_stats(dev);
        }
        if (dev.lib) {
            ggml_metal_library_free(dev.lib);
            dev.lib = nullptr;
        }
    }
    delete session;
}

static void metal_session_enter(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed++;
        return;
    }

    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || session->dormant.load() || session->stopping) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    try {
        g_session_stack.push_back({session, session});
    } catch (...) {
        g_session_suppressed++;
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    session->active_scopes++;
}

static void metal_session_leave(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed--;
        return;
    }
    moe_cache_session * expected = (moe_cache_session *)opaque;
    auto found = std::find_if(
            g_session_stack.rbegin(), g_session_stack.rend(),
            [expected](const moe_cache_scope_frame & frame) {
                return frame.requested == expected;
            });
    if (found == g_session_stack.rend()) {
        return;
    }
    moe_cache_session * active = found->active;
    g_session_stack.erase(std::next(found).base());
    if (active) {
        std::lock_guard<std::mutex> lock(active->mu);
        if (active->active_scopes > 0) {
            active->active_scopes--;
        }
        active->idle_cv.notify_all();
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
        dst[ib].d = ggml_fp32_to_fp16(d);
        dst[ib].s = ggml_fp32_to_fp16(sum);
    }
}

// ---------------------------------------------------------------------------
// Begin: find pool, create node. Pools are created lazily on first use.
// ---------------------------------------------------------------------------

static void * metal_begin(const char * name, const void * host_base,
                           size_t expert_size, int64_t n_in, int64_t n_out,
                           int wtype, int64_t n_expert, int64_t n_tokens,
                           int64_t n_rows) {
    if (g_session_suppressed > 0 || g_session_stack.empty()) {
        return nullptr;
    }
    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant) {
        return nullptr;
    }
    if (!name || !host_base || !moe_cache_tensor_name_supported(name) ||
        n_tokens < 1 || expert_size < session->config.min_expert_bytes ||
        n_in <= 0 || n_out <= 0 || n_expert <= 0 ||
        !ggml_moe_cache_wtype_supported(wtype)) {
        return nullptr;
    }
    if (n_rows < n_tokens || n_rows % n_tokens != 0 ||
        n_tokens > session->config.max_batch ||
        n_rows > moe_cache_node_rows_max) {
        return nullptr;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size != (size_t)n_out * row_size ||
        expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return nullptr;
    }

    if (session->devices.empty()) {
        return nullptr;
    }
    moe_cache_metal_device & dev =
        static_cast<moe_cache_metal_device &>(*session->devices[0]);
    // A zero budget can never create a pool; skip GPU setup entirely.
    if (session->config.budget_mb == 0) {
        return nullptr;
    }
    // Compile the pipelines and create the command queue on first use, before
    // the dispatch lock so a one-time compile does not block other workers.
    if (!metal_device_ensure_ready(dev, session->config.budget_mb)) {
        return nullptr;
    }

    std::unique_lock<std::mutex> dispatch_lock;
    try {
        dispatch_lock = std::unique_lock<std::mutex>(
                dev.dispatch_mu, std::try_to_lock);
    } catch (...) {
        dev.contention_bypasses++;
        return nullptr;
    }
    if (!dispatch_lock.owns_lock()) {
        dev.contention_bypasses++;
        return nullptr;
    }
    if (dev.dead.load()) {
        return nullptr;
    }

    moe_cache_log_configuration(*session);
    const size_t budget_bytes = session->config.budget_mb << 20;
    moe_cache_pool * pool = metal_find_or_create_pool(
            dev, *session, expert_size, wtype, n_expert, budget_bytes);
    if (!pool) {
        return nullptr;
    }
    const int pool_index = moe_cache_find_pool(dev, expert_size, wtype);
    if (pool_index < 0) {
        return nullptr;
    }

    std::unique_ptr<moe_cache_node> node(new (std::nothrow) moe_cache_node());
    if (!node) {
        return nullptr;
    }
    node->session = session;
    node->device = &dev;
    node->pool = pool;
    node->pool_index = pool_index;
    node->host_base = host_base;
    node->expert_size = expert_size;
    node->n_in = n_in;
    node->n_out = n_out;
    node->n_expert = n_expert;
    node->n_tokens = n_tokens;
    node->wtype = wtype;
    node->dispatch_lock = std::move(dispatch_lock);

    // Own the node for the caller; destroy() waits for this to reach zero.
    std::lock_guard<std::mutex> session_lock(session->mu);
    session->active_nodes++;
    return node.release();
}

// ---------------------------------------------------------------------------
// Plan: mark cache hits; sync-fill misses into the slab so they can also
// be served by this node. slot_idx[i] >= 0 means the row is cache-served.
// ---------------------------------------------------------------------------

static int metal_plan(void * opaque, const int32_t * ids, int n_ids,
                       int32_t * slot_indices) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !ids || !slot_indices || n_ids < 0 ||
        n_ids > moe_cache_node_rows_max || node->planned) {
        return 0;
    }
    node->planned = true;
    for (int index = 0; index < n_ids; index++) {
        slot_indices[index] = -1;
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    moe_cache_pool & pool = *node->pool;
    int hits = 0;

    std::unique_lock<std::mutex> lock(session.mu);
    if (session.stopping) {
        return 0;
    }
    for (int index = 0; index < n_ids; index++) {
        const int32_t expert = ids[index];
        if (expert < 0 || expert >= node->n_expert || device.dead.load()) {
            continue;
        }

        const moe_cache_key key{node->host_base, expert};
        auto found = pool.map.find(key);
        if (found != pool.map.end() &&
            pool.slots[found->second].state == moe_cache_slot_state::valid) {
            const int slot_index = found->second;
            moe_cache_slot & slot = pool.slots[slot_index];
            slot.readers++;
            slot.uses++;
            moe_cache_lru_remove(pool, slot_index);
            moe_cache_lru_push_back(pool, slot_index);
            node->pins[node->n_pins++] = {&pool, slot_index};
            slot_indices[index] = slot_index;
            device.hits++;
            hits++;
            continue;
        }

        // miss: evict LRU if full, then sync-fill
        device.misses++;
        if (moe_cache_fail(session, "insert")) {
            device.fill_failures++;
            continue;
        }
        int slot_index = -1;
        if (!pool.free_slots.empty()) {
            slot_index = pool.free_slots.back();
            pool.free_slots.pop_back();
        } else {
            int candidate = moe_cache_pick_victim(pool, session.config.hot_uses);
            if (candidate < 0) {
                continue; // all slots pinned; CPU handles this row
            }
            slot_index = candidate;
            const bool sacrificed_hot =
                (int)pool.slots[slot_index].uses > session.config.hot_uses;
            moe_cache_slot_reset(pool, slot_index, false);
            device.evictions++;
            if (sacrificed_hot) {
                device.heat_evictions++;
            }
        }

        moe_cache_slot & slot = pool.slots[slot_index];
        slot.key = key;
        slot.generation++;
        slot.state = moe_cache_slot_state::copying;
        const void * source =
            (const char *)node->host_base + (size_t)expert * node->expert_size;
        try {
            pool.map.emplace(key, slot_index);
        } catch (...) {
            moe_cache_slot_reset(pool, slot_index, true);
            device.insert_skips++;
            continue;
        }

        // unified memory: direct copy into the shared slab
        memcpy(pool.slab + (size_t)slot_index * node->expert_size,
               source, node->expert_size);
        slot.state = moe_cache_slot_state::valid;
        moe_cache_lru_push_back(pool, slot_index);

        slot.readers++;
        node->pins[node->n_pins++] = {&pool, slot_index};
        slot_indices[index] = slot_index;
        device.inserts++;
        device.fills++;
        device.hits++;
        hits++;
    }
    device.nodes++;
    return hits;
}

// ---------------------------------------------------------------------------
// Dispatch: launch the Metal matvec kernel over the hit rows.
// ---------------------------------------------------------------------------

static int metal_dispatch(void * opaque, int wtype, int64_t n_in, int64_t n_out,
                           int n_hits, const int32_t * slot_indices,
                           const float * const * act_rows) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !node->planned || !slot_indices || !act_rows ||
        n_hits <= 0 || n_hits > moe_cache_node_rows_max ||
        n_hits != node->n_pins ||
        wtype != node->wtype || n_in != node->n_in || n_out != node->n_out) {
        return 0;
    }

    moe_cache_metal_device & dev =
        static_cast<moe_cache_metal_device &>(*node->device);
    if (dev.dead.load() || moe_cache_fail(*node->session, "dispatch")) {
        std::lock_guard<std::mutex> lock(node->session->mu);
        dev.dispatch_failures++;
        return 0;
    }

    const int64_t padded_n_in = ((n_in + QK8_1 - 1) / QK8_1) * QK8_1;
    const int64_t expert_stride = node->expert_size;
    const int64_t row_stride = ggml_row_size((ggml_type)wtype, n_in);

    // Output buffer (shared; CPU reads results directly).
    const size_t out_bytes = (size_t)n_hits * (size_t)n_out * sizeof(float);
    id<MTLBuffer> out_buf = [dev.mtl_device newBufferWithLength:out_bytes
        options:MTLResourceStorageModeShared];
    if (!out_buf) {
        return 0;
    }

    // Activation buffer: q8_1 per hit row.
    const size_t act_bytes = (size_t)n_hits * (size_t)(padded_n_in / QK8_1) *
        sizeof(block_q8_1);
    id<MTLBuffer> act_buf = [dev.mtl_device newBufferWithLength:act_bytes
        options:MTLResourceStorageModeShared];
    if (!act_buf) {
        [out_buf release];
        return 0;
    }
    block_q8_1 * act_q8 = (block_q8_1 *)[act_buf contents];
    for (int i = 0; i < n_hits; i++) {
        metal_quantize_act_q8_1(act_rows[i],
            act_q8 + i * (padded_n_in / QK8_1), n_in, padded_n_in);
    }

    // Slot index buffer.
    const size_t ids_bytes = (size_t)n_hits * sizeof(int32_t);
    id<MTLBuffer> ids_buf = [dev.mtl_device newBufferWithLength:ids_bytes
        options:MTLResourceStorageModeShared];
    if (!ids_buf) {
        [act_buf release];
        [out_buf release];
        return 0;
    }
    memcpy([ids_buf contents], slot_indices, ids_bytes);

    // Select pipeline by weight type.
    struct ggml_metal_pipeline_with_params pipeline = dev.mmv_pipeline_q8_0;
    switch (wtype) {
        case GGML_TYPE_Q8_0: pipeline = dev.mmv_pipeline_q8_0; break;
        case GGML_TYPE_Q4_0: pipeline = dev.mmv_pipeline_q4_0; break;
        case GGML_TYPE_Q4_K: pipeline = dev.mmv_pipeline_q4_K; break;
        case GGML_TYPE_Q6_K: pipeline = dev.mmv_pipeline_q6_K; break;
        case GGML_TYPE_Q5_K: pipeline = dev.mmv_pipeline_q5_K; break;
        case GGML_TYPE_Q1_0: pipeline = dev.mmv_pipeline_q1_0; break;
        case GGML_TYPE_Q2_0: pipeline = dev.mmv_pipeline_q2_0; break;
        case GGML_TYPE_Q4_1: pipeline = dev.mmv_pipeline_q4_1; break;
        case GGML_TYPE_Q5_0: pipeline = dev.mmv_pipeline_q5_0; break;
        case GGML_TYPE_Q5_1: pipeline = dev.mmv_pipeline_q5_1; break;
        case GGML_TYPE_Q2_K: pipeline = dev.mmv_pipeline_q2_K; break;
        case GGML_TYPE_Q3_K: pipeline = dev.mmv_pipeline_q3_K; break;
        case GGML_TYPE_IQ2_XXS: pipeline = dev.mmv_pipeline_iq2_xxs; break;
        case GGML_TYPE_IQ2_XS: pipeline = dev.mmv_pipeline_iq2_xs; break;
        case GGML_TYPE_IQ2_S: pipeline = dev.mmv_pipeline_iq2_s; break;
        case GGML_TYPE_IQ3_XXS: pipeline = dev.mmv_pipeline_iq3_xxs; break;
        case GGML_TYPE_IQ3_S: pipeline = dev.mmv_pipeline_iq3_s; break;
        case GGML_TYPE_IQ1_S: pipeline = dev.mmv_pipeline_iq1_s; break;
        case GGML_TYPE_IQ1_M: pipeline = dev.mmv_pipeline_iq1_m; break;
        case GGML_TYPE_IQ4_NL: pipeline = dev.mmv_pipeline_iq4_nl; break;
        case GGML_TYPE_IQ4_XS: pipeline = dev.mmv_pipeline_iq4_xs; break;
        case GGML_TYPE_MXFP4: pipeline = dev.mmv_pipeline_mxfp4; break;
        case GGML_TYPE_NVFP4: pipeline = dev.mmv_pipeline_nvfp4; break;
        default: break;
    }
    if (!pipeline.pipeline) {
        [ids_buf release];
        [act_buf release];
        [out_buf release];
        return 0;
    }

    // Find the slab MTLBuffer that backs this pool.
    id<MTLBuffer> slab_buf = nil;
    NSUInteger slab_offset = 0;
    for (id<MTLBuffer> buf : dev.slab_buffers) {
        char * contents = (char *)[buf contents];
        char * slab = node->pool->slab;
        if (slab >= contents && slab < contents + [buf length]) {
            slab_buf = buf;
            slab_offset = (NSUInteger)(slab - contents);
            break;
        }
    }
    if (!slab_buf) {
        [ids_buf release];
        [act_buf release];
        [out_buf release];
        return 0;
    }

    // MTLCommandQueue has no -newCommandBuffer; the selector is -commandBuffer,
    // which returns an autoreleased buffer (+0), so it must not be released here.
    id<MTLCommandBuffer> cmd_buf = [dev.mtl_queue commandBuffer];
    if (!cmd_buf) {
        [ids_buf release];
        [act_buf release];
        [out_buf release];
        return 0;
    }
    ggml_metal_encoder_t enc = ggml_metal_encoder_init(
            (ggml_metal_cmd_buf_t)cmd_buf, true);
    if (!enc) {
        [ids_buf release];
        [act_buf release];
        [out_buf release];
        return 0;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer(enc, {(__bridge void *)slab_buf, slab_offset}, 0);
    ggml_metal_encoder_set_buffer(enc, {(__bridge void *)ids_buf, 0}, 1);
    ggml_metal_encoder_set_buffer(enc, {(__bridge void *)act_buf, 0}, 2);
    ggml_metal_encoder_set_buffer(enc, {(__bridge void *)out_buf, 0}, 3);

    int64_t args[6] = {n_in, n_out, expert_stride, row_stride, n_hits, padded_n_in};
    ggml_metal_encoder_set_bytes(enc, args, sizeof(args), 4);

    const NSUInteger total_threads = (NSUInteger)(n_hits * n_out);
    ggml_metal_encoder_dispatch_threadgroups(
            enc, (int)((total_threads + 255) / 256), 1, 1, 256, 1, 1);

    ggml_metal_encoder_end_encoding(enc);
    ggml_metal_encoder_free(enc);

    [cmd_buf commit];
    [cmd_buf waitUntilCompleted];

    // A failed command buffer leaves out_buf with undefined contents. Do not
    // publish it: report failure so the CPU path recomputes these rows.
    const bool cmd_ok = [cmd_buf status] == MTLCommandBufferStatusCompleted;
    if (!cmd_ok) {
        NSError * mtl_err = [cmd_buf error];
        MOE_CACHE_LOG("[moe-cache] Metal%d: command buffer failed (status=%ld, error=%s); "
                      "dropping the matvec so the CPU path recomputes\n",
                      dev.physical, (long)[cmd_buf status],
                      mtl_err ? [[mtl_err localizedDescription] UTF8String] : "none");
    }

    [ids_buf release];
    [act_buf release];
    if (!cmd_ok) {
        [out_buf release];
        dev.dispatch_failures++;
        return 0;
    }

    if (dev.out_buffer) {
        [dev.out_buffer release];
    }
    dev.out_buffer = out_buf;
    dev.d_out = [out_buf contents];
    dev.d_out_cap = out_bytes;

    node->dispatched = true;
    return 1;
}

// ---------------------------------------------------------------------------
// Collect: copy results from the shared output buffer into dst_rows.
// ---------------------------------------------------------------------------

static int metal_collect(void * opaque, int n_hits, float * const * dst_rows,
                          int64_t n_out) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !node->dispatched || n_hits <= 0 ||
        n_hits > moe_cache_node_rows_max ||
        node->n_pins != n_hits || !dst_rows || n_out != node->n_out) {
        return 0;
    }
    for (int index = 0; index < n_hits; index++) {
        if (!dst_rows[index]) {
            return 0;
        }
    }

    moe_cache_metal_device & dev =
        static_cast<moe_cache_metal_device &>(*node->device);
    moe_cache_session & session = *node->session;
    const bool ok = !dev.dead.load() && dev.out_buffer && dev.d_out &&
        !moe_cache_fail(session, "collect");
    if (ok) {
        float * out = (float *)dev.d_out;
        for (int index = 0; index < n_hits; index++) {
            memcpy(dst_rows[index], out + (size_t)index * n_out,
                   (size_t)n_out * sizeof(float));
        }
    }
    node->dispatched = false;
    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (!ok) {
            dev.collect_failures++;
        }
        dev.collect_calls++;
        if (session.config.stats_every > 0 &&
            dev.collect_calls % session.config.stats_every == 0) {
            metal_log_stats(dev);
        }
    }
    return ok ? 1 : 0;
}

// ---------------------------------------------------------------------------
// End: release slot pins and the node.
// ---------------------------------------------------------------------------

static void metal_end(void * opaque) {
    std::unique_ptr<moe_cache_node> node((moe_cache_node *)opaque);
    if (!node) {
        return;
    }
    moe_cache_session & session = *node->session;
    {
        std::lock_guard<std::mutex> lock(session.mu);
        for (int index = 0; index < node->n_pins; index++) {
            const moe_cache_pin & pin = node->pins[index];
            if (pin.pool && pin.slot >= 0 && pin.slot < pin.pool->n_slots) {
                moe_cache_slot & slot = pin.pool->slots[pin.slot];
                if (slot.readers > 0) {
                    slot.readers--;
                }
            }
        }
        session.active_nodes--;
        session.idle_cv.notify_all();
    }
}

// ---------------------------------------------------------------------------
// Fused SwiGLU — not implemented for v1; stock CPU path handles the node.
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

// ---------------------------------------------------------------------------
// Invalidate: drop cached slots whose tensor range overlaps [base, base+size).
// ---------------------------------------------------------------------------

static void metal_invalidate(const void * base, size_t size) {
    if (!base || size == 0) {
        return;
    }
    if (g_session_count.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        std::lock_guard<std::mutex> lock(session->mu);
        for (auto & dev_ptr : session->devices) {
            for (auto & pool_ptr : dev_ptr->pools) {
                for (int i = 0; i < pool_ptr->n_slots; i++) {
                    if (pool_ptr->slots[i].key.tensor &&
                        moe_cache_ranges_overlap(
                            pool_ptr->slots[i].key.tensor,
                            pool_ptr->expert_size, base, size)) {
                        moe_cache_slot_reset(*pool_ptr, i, true);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

// Each backend wires its own function table into the provider registry
// (ggml-backend-moe-cache.h). The owner is the backend reg pointer so that
// query_device can match ggml_backend_dev_backend_reg(device).

static void metal_register(const void * owner) {
    g_moe_cache_owner = owner;
    ggml_moe_cache_api api = {};
    api.owner = owner;
    api.query_config = metal_query_config;
    api.query_device = metal_query_device;
    api.query_shape = metal_query_shape;
    api.session_create = metal_session_create;
    api.session_destroy = metal_session_destroy;
    api.session_enter = metal_session_enter;
    api.session_leave = metal_session_leave;
    api.begin = metal_begin;
    api.plan = metal_plan;
    api.dispatch = metal_dispatch;
    api.collect = metal_collect;
    api.end = metal_end;
    api.fused_begin = metal_fused_begin;
    api.invalidate = metal_invalidate;
    ggml_moe_cache_register(&api);
}

// Prior declaration so the definition below does not trip -Wmissing-prototypes.
extern "C" void ggml_metal_moe_cache_register(void * reg);

extern "C" void ggml_metal_moe_cache_register(void * reg) {
    metal_register(reg);
}
