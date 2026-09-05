#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class llama_kv_stream_region_role : uint8_t {
    target,
    mtp,
};

// Smallest independently resident piece of a logical KV cache. The CUDA copy
// layer may coalesce adjacent regions into a larger transfer slot.
struct llama_kv_stream_region {
    llama_kv_stream_region_role role = llama_kv_stream_region_role::target;

    int32_t  layer_id    = -1;
    uint32_t token_begin = 0;
    uint32_t token_count = 0;

    uint64_t bytes = 0;

    // Lower values have stronger residency preference. The ordering must be
    // stable across replans to avoid unnecessary promotion/demotion churn.
    uint64_t residency_priority = 0;

    // Pinned is a logical pool policy, not cudaMalloc pinning. A pinned region
    // must receive permanent pool space or planning fails.
    bool pinned = false;
};

struct llama_kv_stream_plan_params {
    uint64_t pool_bytes       = 0;
    uint64_t stage_slot_bytes = 0;
    uint32_t stage_slots      = 0;

    std::vector<llama_kv_stream_region> regions;
};

struct llama_kv_stream_plan {
    bool valid = false;
    std::string error;

    uint64_t pool_bytes           = 0;
    uint64_t reserved_stage_bytes = 0;
    uint64_t resident_bytes       = 0;
    uint64_t streamed_bytes       = 0;
    uint64_t unused_bytes         = 0;

    std::vector<size_t> resident_regions;
    std::vector<size_t> streamed_regions;
};

llama_kv_stream_plan llama_kv_stream_plan_make(const llama_kv_stream_plan_params & params);

struct llama_kv_stream_extent_params {
    uint32_t live_tokens              = 0;
    uint32_t reserve_tokens           = 0;
    uint32_t page_tokens              = 256;
    uint32_t previous_extent          = 0;
    uint32_t shrink_hysteresis_tokens = 0;
    uint32_t maximum_tokens           = 0;

    bool force_shrink = false;
};

struct llama_kv_stream_extent {
    bool valid = false;
    std::string error;

    uint32_t tokens = 0;
    bool grew   = false;
    bool shrunk = false;
};

llama_kv_stream_extent llama_kv_stream_extent_make(const llama_kv_stream_extent_params & params);

struct llama_kv_stream_layer_layout {
    llama_kv_stream_region_role role = llama_kv_stream_region_role::target;

    int32_t  layer_id        = -1;
    uint32_t n_tokens        = 0;
    uint64_t bytes_per_token = 0;

    // Lower values spread residency preference within the same token page.
    uint32_t layer_priority = 0;

    bool pin_all  = false;
    bool pin_tail = false;
};

struct llama_kv_stream_regions_params {
    uint32_t page_tokens = 256;
    std::vector<llama_kv_stream_layer_layout> layers;
};

struct llama_kv_stream_regions {
    bool valid = false;
    std::string error;

    uint64_t total_bytes = 0;
    std::vector<llama_kv_stream_region> regions;
};

llama_kv_stream_regions llama_kv_stream_regions_make(const llama_kv_stream_regions_params & params);

enum class llama_kv_stream_prefetch_state : uint8_t {
    pending,
    scheduled,
    consumed,
};

struct llama_kv_stream_prefetch_request {
    int32_t  layer_id        = -1;
    uint32_t attention_index = 0;
    uint32_t page_index      = 0;
    uint64_t bytes           = 0;

    // -1 means immutable at graph start. A non-negative value identifies the
    // attention step whose SET_ROWS operation must finish before this page may
    // be copied (normally the mutable tail page).
    int32_t producer_attention_index = -1;
};

struct llama_kv_stream_prefetch_assignment {
    size_t request_index = 0;
    uint32_t slot        = 0;
};

struct llama_kv_stream_prefetch_params {
    uint32_t current_attention = 0;
    uint32_t lookahead_layers  = 0;
    uint64_t stage_slot_bytes  = 0;

    bool adaptive_lookahead = false;
    bool current_producer_complete = false;

    std::vector<uint32_t> free_slots;
    std::vector<llama_kv_stream_prefetch_request> requests;
    std::vector<llama_kv_stream_prefetch_state> states;
};

struct llama_kv_stream_prefetch_dispatch_result {
    bool valid = false;
    std::string error;
    std::vector<llama_kv_stream_prefetch_assignment> assignments;
};

// Selects work for the currently free transfer slots. The caller owns the
// asynchronous lifecycle and changes request states only after recording the
// corresponding CUDA ready/consumed events.
llama_kv_stream_prefetch_dispatch_result llama_kv_stream_prefetch_dispatch(
    const llama_kv_stream_prefetch_params & params);

struct llama_kv_stream_partition_params {
    uint32_t total_pool_pages       = 0;
    uint32_t layer_count            = 0;
    uint32_t active_pages_per_layer = 0;
    uint32_t minimum_ring_slots     = 0;

    uint32_t previous_resident_pages_per_layer = 0;
    uint32_t previous_ring_slots               = 0;

    double deadline_miss_ratio       = 0.0;
    double copy_engine_busy_ratio    = 0.0;
    double ring_peak_occupancy_ratio = 0.0;

    // Ring slots relative to the streamed pages consumed by one attention
    // layer. A value above one leaves room to begin the next layer early.
    double target_ring_working_set_ratio = 1.10;

    uint32_t starved_evaluations        = 0;
    uint32_t overprovisioned_evaluations = 0;
    uint32_t grow_hysteresis_evaluations = 3;
    uint32_t shrink_hysteresis_evaluations = 8;
    uint32_t evaluations_since_repartition = UINT32_MAX;
    uint32_t repartition_cooldown_evaluations = 64;

    // The first decode graph has no useful streaming feedback yet, but its
    // active working set is already known. Select the deterministic overlap
    // target immediately instead of spending several tokens in an undersized
    // prefill ring before hysteresis can react.
    bool entering_decode_layout = false;
};

struct llama_kv_stream_partition {
    bool valid = false;
    std::string error;
    bool changed = false;

    uint32_t resident_pages_per_layer = 0;
    uint32_t ring_slots               = 0;
    uint32_t starved_evaluations      = 0;
    uint32_t overprovisioned_evaluations = 0;
};

// Adjusts the boundary inside a fixed page pool. A growth step demotes exactly
// one page from every layer, turning those addresses into immediately reusable
// ring slots. Promotion performs the inverse operation after longer hysteresis.
llama_kv_stream_partition llama_kv_stream_partition_adapt(
    const llama_kv_stream_partition_params & params);

struct llama_kv_stream_feedback_counters {
    uint64_t deadline_samples = 0;
    uint64_t deadline_misses  = 0;
};

struct llama_kv_stream_feedback_delta {
    bool valid = false;
    bool has_evaluation = false;
    std::string error;
    uint64_t deadline_samples = 0;
    uint64_t deadline_misses  = 0;
    double deadline_miss_ratio = 0.0;
};

llama_kv_stream_feedback_delta llama_kv_stream_feedback_delta_make(
    const llama_kv_stream_feedback_counters & current,
    const llama_kv_stream_feedback_counters & previous);
