#include "llama-kv-stream-config.h"

#include <limits>

llama_kv_stream_config_result llama_kv_stream_config_validate(const llama_kv_stream_config & config) {
    if (config.arena_bytes == 0) {
        return { true, false, {} };
    }

    auto invalid = [](const char * error) {
        return llama_kv_stream_config_result { false, false, error };
    };

    if (!config.unified_kv_cache) {
        return invalid("block KV streaming requires a standard unified KV cache "
                        "(unsupported on MLA/indexer or SWA dual-cache architectures)");
    }
    if (!config.context_default) {
        return invalid("block KV streaming currently supports only the target context, not MTP/draft contexts");
    }
    if (!config.single_sequence) {
        return invalid("block KV streaming requires exactly one sequence (-np 1)");
    }
    if (!config.flash_attention) {
        return invalid("block KV streaming requires Flash Attention");
    }
    if (!config.kv_offload) {
        return invalid("block KV streaming requires GPU KV offload");
    }
    if (!config.type_pair_supported) {
        return invalid("block KV streaming is not supported for this K/V cache type pair on the active backend");
    }
    if (config.minimum_arena_bytes == 0 || config.arena_bytes < config.minimum_arena_bytes) {
        return invalid("block KV streaming arena is too small");
    }

    return { true, true, {} };
}

llama_kv_stream_pool_layout llama_kv_stream_pool_layout_make(
        const llama_kv_stream_pool_layout_params & params) {
    llama_kv_stream_pool_layout result;

    auto invalid = [&](const char * error) {
        result.error = error;
        return result;
    };

    if (params.pool_bytes == 0 || params.page_bytes == 0 ||
            params.layer_count == 0 || params.scratch_pages == 0) {
        return invalid("pool, page, layer, and scratch counts must be nonzero");
    }
    if (params.page_bytes > std::numeric_limits<uint64_t>::max()/params.scratch_pages) {
        return invalid("scratch byte count overflow");
    }
    result.scratch_bytes = params.page_bytes*params.scratch_pages;
    if (result.scratch_bytes >= params.pool_bytes) {
        return invalid("pool has no resident capacity after reserving scratch pages");
    }
    if (params.page_bytes > std::numeric_limits<uint64_t>::max()/params.layer_count) {
        return invalid("per-layer partition byte count overflow");
    }

    const uint64_t bytes_per_round = params.page_bytes*params.layer_count;
    const uint64_t pages = (params.pool_bytes - result.scratch_bytes)/bytes_per_round;
    if (pages == 0 || pages > std::numeric_limits<uint32_t>::max()/256U) {
        return invalid("pool cannot hold one resident page per layer");
    }

    result.resident_pages_per_layer  = pages;
    result.resident_tokens_per_layer = pages*256U;
    result.resident_bytes = pages*bytes_per_round;
    result.unused_bytes = params.pool_bytes - result.scratch_bytes - result.resident_bytes;
    result.valid = true;
    return result;
}

llama_kv_stream_phase_plan llama_kv_stream_phase_plan_make(
        const llama_kv_stream_phase_plan_params & params) {
    llama_kv_stream_phase_plan result;

    auto invalid = [&](const char * error) {
        result.error = error;
        return result;
    };

    if (params.arena_bytes == 0 || params.compute_bytes == 0 ||
            params.compute_alignment == 0 || params.page_bytes == 0 ||
            params.layer_count == 0 || params.minimum_ring_pages == 0) {
        return invalid("arena, compute, alignment, page, layer, and ring counts must be nonzero");
    }
    if (params.compute_bytes >= params.arena_bytes) {
        return invalid("arena has no KV capacity after reserving compute space");
    }

    const uint64_t unaligned_compute_offset = params.arena_bytes - params.compute_bytes;
    result.compute_offset =
        (unaligned_compute_offset/params.compute_alignment)*params.compute_alignment;
    result.compute_bytes = params.arena_bytes - result.compute_offset;
    result.kv_offset = 0;
    result.kv_bytes = result.compute_offset;

    if (params.page_bytes > std::numeric_limits<uint64_t>::max()/params.minimum_ring_pages) {
        return invalid("ring byte count overflow");
    }
    result.conversion_bytes = params.conversion_bytes;

    const uint64_t minimum_ring_bytes =
        params.page_bytes*params.minimum_ring_pages;
    if (result.conversion_bytes > result.kv_bytes ||
            minimum_ring_bytes > result.kv_bytes - result.conversion_bytes) {
        return invalid("KV space cannot hold conversion and ring reservations");
    }
    if (params.page_bytes > std::numeric_limits<uint64_t>::max()/params.layer_count) {
        return invalid("per-layer partition byte count overflow");
    }

    const uint64_t bytes_per_round = params.page_bytes*params.layer_count;
    const uint64_t page_budget = result.kv_bytes - result.conversion_bytes;
    const uint64_t total_pages = page_budget/params.page_bytes;
    const uint64_t resident_pages_per_layer =
        (total_pages - params.minimum_ring_pages)/params.layer_count;
    if (resident_pages_per_layer == 0 ||
            resident_pages_per_layer > std::numeric_limits<uint32_t>::max()) {
        return invalid("KV space cannot hold one resident page per layer");
    }

    const uint64_t ring_pages =
        total_pages - resident_pages_per_layer*params.layer_count;
    if (ring_pages < params.minimum_ring_pages ||
            ring_pages > std::numeric_limits<uint32_t>::max()) {
        return invalid("runtime ring page count exceeds supported range");
    }

    result.resident_pages_per_layer = resident_pages_per_layer;
    result.ring_pages = ring_pages;
    result.resident_bytes = resident_pages_per_layer*bytes_per_round;
    result.ring_bytes = ring_pages*params.page_bytes;
    result.unused_bytes =
        result.kv_bytes - result.conversion_bytes - result.ring_bytes - result.resident_bytes;
    result.valid = true;
    return result;
}

bool llama_kv_stream_phase_is_generation(
        llama_kv_stream_phase phase, uint32_t batch_tokens) {
    switch (phase) {
        case LLAMA_KV_STREAM_PHASE_AUTOMATIC:
            return batch_tokens == 1;
        case LLAMA_KV_STREAM_PHASE_PROMPT:
            return false;
        case LLAMA_KV_STREAM_PHASE_GENERATION:
            return true;
    }
    return batch_tokens == 1;
}
