#include "llama-kv-stream-plan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>

namespace {

struct logical_interval {
    llama_kv_stream_region_role role;
    int32_t  layer_id;
    uint64_t begin;
    uint64_t end;
};

bool checked_add(uint64_t a, uint64_t b, uint64_t & result) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }

    result = a + b;
    return true;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t & result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max()/a) {
        return false;
    }

    result = a*b;
    return true;
}

llama_kv_stream_plan fail(llama_kv_stream_plan result, const char * message) {
    result.valid = false;
    result.error = message;
    return result;
}

} // namespace

llama_kv_stream_plan llama_kv_stream_plan_make(const llama_kv_stream_plan_params & params) {
    llama_kv_stream_plan result;
    result.pool_bytes = params.pool_bytes;

    if (params.pool_bytes == 0) {
        return fail(std::move(result), "KV stream pool must be non-zero");
    }

    if (params.stage_slots > 0 && params.stage_slot_bytes == 0) {
        return fail(std::move(result), "non-zero stage count requires a non-zero stage size");
    }

    if (!checked_mul(params.stage_slots, params.stage_slot_bytes, result.reserved_stage_bytes)) {
        return fail(std::move(result), "stage reservation size overflow");
    }

    if (result.reserved_stage_bytes > params.pool_bytes) {
        return fail(std::move(result), "stage reservations exceed the KV stream pool");
    }

    std::vector<size_t> optional_regions;
    std::vector<logical_interval> logical_intervals;
    optional_regions.reserve(params.regions.size());
    logical_intervals.reserve(params.regions.size());
    result.resident_regions.reserve(params.regions.size());
    result.streamed_regions.reserve(params.regions.size());

    uint64_t pinned_bytes = 0;

    for (size_t index = 0; index < params.regions.size(); ++index) {
        const auto & region = params.regions[index];

        if (region.layer_id < 0) {
            return fail(std::move(result), "KV stream region has an invalid layer id");
        }

        if (region.token_count == 0 || region.bytes == 0) {
            return fail(std::move(result), "KV stream region must contain tokens and bytes");
        }

        const uint64_t token_end = uint64_t(region.token_begin) + region.token_count;
        if (token_end > uint64_t(std::numeric_limits<uint32_t>::max()) + 1) {
            return fail(std::move(result), "KV stream region token range overflow");
        }

        logical_intervals.push_back({ region.role, region.layer_id, region.token_begin, token_end });

        if (region.pinned) {
            if (!checked_add(pinned_bytes, region.bytes, pinned_bytes)) {
                return fail(std::move(result), "pinned KV region size overflow");
            }
            result.resident_regions.push_back(index);
        } else {
            optional_regions.push_back(index);
        }
    }

    std::sort(logical_intervals.begin(), logical_intervals.end(), [](const auto & lhs, const auto & rhs) {
        return std::tie(lhs.role, lhs.layer_id, lhs.begin, lhs.end) <
               std::tie(rhs.role, rhs.layer_id, rhs.begin, rhs.end);
    });

    for (size_t i = 1; i < logical_intervals.size(); ++i) {
        const auto & previous = logical_intervals[i - 1];
        const auto & current  = logical_intervals[i];

        if (previous.role == current.role &&
            previous.layer_id == current.layer_id &&
            current.begin < previous.end) {
            return fail(std::move(result), "KV stream regions overlap in one logical layer");
        }
    }

    uint64_t required_bytes = 0;
    if (!checked_add(result.reserved_stage_bytes, pinned_bytes, required_bytes)) {
        return fail(std::move(result), "mandatory KV stream pool size overflow");
    }

    if (required_bytes > params.pool_bytes) {
        return fail(std::move(result), "pinned KV regions and stages exceed the pool");
    }

    result.resident_bytes = pinned_bytes;
    uint64_t remaining_bytes = params.pool_bytes - required_bytes;

    std::sort(optional_regions.begin(), optional_regions.end(), [&](size_t a, size_t b) {
        const auto & lhs = params.regions[a];
        const auto & rhs = params.regions[b];

        return std::tie(lhs.residency_priority, lhs.role, lhs.layer_id, lhs.token_begin, a) <
               std::tie(rhs.residency_priority, rhs.role, rhs.layer_id, rhs.token_begin, b);
    });

    for (size_t index : optional_regions) {
        const auto & region = params.regions[index];

        if (region.bytes <= remaining_bytes) {
            result.resident_regions.push_back(index);
            result.resident_bytes += region.bytes;
            remaining_bytes -= region.bytes;
            continue;
        }

        if (params.stage_slots == 0) {
            return fail(std::move(result), "streaming is required but no stage slots are configured");
        }

        if (region.bytes > params.stage_slot_bytes) {
            return fail(std::move(result), "a streamed KV region exceeds the stage slot size");
        }

        uint64_t streamed_bytes = 0;
        if (!checked_add(result.streamed_bytes, region.bytes, streamed_bytes)) {
            return fail(std::move(result), "streamed KV region size overflow");
        }

        result.streamed_bytes = streamed_bytes;
        result.streamed_regions.push_back(index);
    }

    result.unused_bytes = remaining_bytes;
    result.valid = true;
    return result;
}

llama_kv_stream_extent llama_kv_stream_extent_make(const llama_kv_stream_extent_params & params) {
    llama_kv_stream_extent result;

    auto fail_extent = [&](const char * message) {
        result.valid = false;
        result.error = message;
        return result;
    };

    if (params.page_tokens == 0) {
        return fail_extent("KV stream page size must be non-zero");
    }

    if (params.maximum_tokens == 0 || params.maximum_tokens%params.page_tokens != 0) {
        return fail_extent("KV stream maximum must be non-zero and page aligned");
    }

    if (params.previous_extent > params.maximum_tokens ||
        params.previous_extent%params.page_tokens != 0) {
        return fail_extent("previous KV stream extent is invalid");
    }

    const uint64_t requested_tokens = uint64_t(params.live_tokens) + params.reserve_tokens;
    if (requested_tokens > params.maximum_tokens) {
        return fail_extent("live and reserved KV tokens exceed the configured maximum");
    }

    uint64_t desired_tokens = 0;
    if (requested_tokens > 0) {
        desired_tokens = ((requested_tokens + params.page_tokens - 1)/params.page_tokens)*params.page_tokens;
    }

    if (desired_tokens > params.maximum_tokens) {
        return fail_extent("padded KV stream extent exceeds the configured maximum");
    }

    uint32_t selected_tokens = uint32_t(desired_tokens);
    if (!params.force_shrink && params.previous_extent > selected_tokens) {
        const uint32_t released_tokens = params.previous_extent - selected_tokens;
        if (released_tokens < params.shrink_hysteresis_tokens) {
            selected_tokens = params.previous_extent;
        }
    }

    result.valid  = true;
    result.tokens = selected_tokens;
    result.grew   = selected_tokens > params.previous_extent;
    result.shrunk = selected_tokens < params.previous_extent;
    return result;
}

llama_kv_stream_regions llama_kv_stream_regions_make(const llama_kv_stream_regions_params & params) {
    llama_kv_stream_regions result;

    auto fail_regions = [&](const char * message) {
        result.valid = false;
        result.error = message;
        result.regions.clear();
        result.total_bytes = 0;
        return result;
    };

    if (params.page_tokens == 0) {
        return fail_regions("KV stream region page size must be non-zero");
    }

    std::set<std::pair<llama_kv_stream_region_role, int32_t>> logical_layers;

    for (const auto & layer : params.layers) {
        if (layer.layer_id < 0) {
            return fail_regions("KV stream layer layout has an invalid layer id");
        }

        if (!logical_layers.emplace(layer.role, layer.layer_id).second) {
            return fail_regions("KV stream layer layout is duplicated");
        }

        if (layer.n_tokens > 0 && layer.bytes_per_token == 0) {
            return fail_regions("non-empty KV stream layer must have a non-zero token size");
        }

        for (uint64_t token_begin = 0; token_begin < layer.n_tokens; token_begin += params.page_tokens) {
            const uint32_t token_count = uint32_t(std::min<uint64_t>(
                params.page_tokens, uint64_t(layer.n_tokens) - token_begin));

            llama_kv_stream_region region;
            region.role        = layer.role;
            region.layer_id    = layer.layer_id;
            region.token_begin = uint32_t(token_begin);
            region.token_count = token_count;
            region.pinned      = layer.pin_all ||
                (layer.pin_tail && token_begin + token_count == layer.n_tokens);

            if (!checked_mul(token_count, layer.bytes_per_token, region.bytes)) {
                return fail_regions("KV stream region byte size overflow");
            }

            uint64_t total_bytes = 0;
            if (!checked_add(result.total_bytes, region.bytes, total_bytes)) {
                return fail_regions("KV stream layout total byte size overflow");
            }
            result.total_bytes = total_bytes;

            const uint64_t page_index = token_begin/params.page_tokens;
            region.residency_priority = (page_index << 32) | layer.layer_priority;
            result.regions.push_back(region);
        }
    }

    result.valid = true;
    return result;
}

llama_kv_stream_prefetch_dispatch_result llama_kv_stream_prefetch_dispatch(
        const llama_kv_stream_prefetch_params & params) {
    llama_kv_stream_prefetch_dispatch_result result;

    auto fail_dispatch = [&](const char * message) {
        result.valid = false;
        result.error = message;
        result.assignments.clear();
        return result;
    };

    if (params.stage_slot_bytes == 0) {
        return fail_dispatch("KV prefetch stage slot size must be non-zero");
    }
    if (params.states.size() != params.requests.size()) {
        return fail_dispatch("KV prefetch request states do not match requests");
    }

    std::vector<uint32_t> slots = params.free_slots;
    std::sort(slots.begin(), slots.end());
    if (std::adjacent_find(slots.begin(), slots.end()) != slots.end()) {
        return fail_dispatch("KV prefetch free slot list contains duplicates");
    }

    const uint64_t window_end = params.adaptive_lookahead ?
        std::numeric_limits<uint32_t>::max() :
        uint64_t(params.current_attention) + params.lookahead_layers;
    std::vector<size_t> eligible;
    eligible.reserve(params.requests.size());

    for (size_t index = 0; index < params.requests.size(); ++index) {
        const auto & request = params.requests[index];
        if (request.layer_id < 0 || request.bytes == 0) {
            return fail_dispatch("KV prefetch request is invalid");
        }
        if (request.bytes > params.stage_slot_bytes) {
            return fail_dispatch("KV prefetch request exceeds a stage slot");
        }
        if (params.states[index] != llama_kv_stream_prefetch_state::pending) {
            continue;
        }
        if (request.attention_index < params.current_attention ||
            uint64_t(request.attention_index) > window_end) {
            continue;
        }

        if (request.producer_attention_index >= 0) {
            const uint32_t producer = uint32_t(request.producer_attention_index);
            if (producer > params.current_attention ||
                (producer == params.current_attention && !params.current_producer_complete)) {
                continue;
            }
        }
        eligible.push_back(index);
    }

    std::sort(eligible.begin(), eligible.end(), [&](size_t a, size_t b) {
        const auto & lhs = params.requests[a];
        const auto & rhs = params.requests[b];
        return std::tie(lhs.attention_index, lhs.page_index, lhs.layer_id, a) <
               std::tie(rhs.attention_index, rhs.page_index, rhs.layer_id, b);
    });

    const size_t count = std::min(slots.size(), eligible.size());
    result.assignments.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.assignments.push_back({ eligible[i], slots[i] });
    }

    result.valid = true;
    return result;
}

llama_kv_stream_partition llama_kv_stream_partition_adapt(
        const llama_kv_stream_partition_params & params) {
    llama_kv_stream_partition result;
    result.resident_pages_per_layer = params.previous_resident_pages_per_layer;
    result.ring_slots = params.previous_ring_slots;

    auto fail_partition = [&](const char * message) {
        result.valid = false;
        result.error = message;
        return result;
    };

    if (params.total_pool_pages == 0 || params.layer_count == 0) {
        return fail_partition("KV stream partition geometry must be non-zero");
    }
    if (params.grow_hysteresis_evaluations == 0 ||
        params.shrink_hysteresis_evaluations == 0 ||
        params.repartition_cooldown_evaluations == 0) {
        return fail_partition("KV stream partition hysteresis must be non-zero");
    }
    const auto ratio_valid = [](double value) {
        return value >= 0.0 && value <= 1.0;
    };
    if (!ratio_valid(params.deadline_miss_ratio) ||
        !ratio_valid(params.copy_engine_busy_ratio) ||
        !ratio_valid(params.ring_peak_occupancy_ratio)) {
        return fail_partition("KV stream partition metrics must be ratios");
    }
    if (!std::isfinite(params.target_ring_working_set_ratio) ||
            params.target_ring_working_set_ratio <= 0.0) {
        return fail_partition("KV stream target working-set ratio must be positive");
    }
    if (params.previous_resident_pages_per_layer > params.active_pages_per_layer) {
        return fail_partition("resident KV pages exceed active pages");
    }

    uint64_t resident_pages = 0;
    if (!checked_mul(params.previous_resident_pages_per_layer, params.layer_count, resident_pages) ||
        resident_pages + params.previous_ring_slots != params.total_pool_pages) {
        return fail_partition("previous KV stream partition does not cover the fixed pool");
    }
    if (params.previous_ring_slots < params.minimum_ring_slots) {
        return fail_partition("previous KV stream ring is below its minimum");
    }

    constexpr double MISS_THRESHOLD = 0.01;
    // A repartition invalidates compact resident addresses. Do not pay that
    // transition cost when the copy engine already has too little headroom
    // for a larger lookahead ring to increase sustained throughput.
    constexpr double COPY_SATURATED  = 0.80;
    constexpr double COPY_LIGHT      = 0.50;
    constexpr double RING_LIGHT      = 0.50;
    constexpr uint32_t FEEDBACK_GROWTH_EPOCHS = 1;

    uint32_t target_resident_pages = 0;
    const uint32_t maximum_resident_pages = std::min(
        params.active_pages_per_layer,
        (params.total_pool_pages - params.minimum_ring_slots)/params.layer_count);
    for (uint32_t resident = maximum_resident_pages;; --resident) {
        const uint32_t ring = params.total_pool_pages - resident*params.layer_count;
        const uint32_t streamed = params.active_pages_per_layer - resident;
        if (streamed == 0 || double(ring) >=
                params.target_ring_working_set_ratio*double(streamed)) {
            target_resident_pages = resident;
            break;
        }
        if (resident == 0) {
            break;
        }
    }
    const uint32_t feedback_resident_floor = target_resident_pages >
        FEEDBACK_GROWTH_EPOCHS ? target_resident_pages - FEEDBACK_GROWTH_EPOCHS : 0;
    const bool overgrown_ring =
        params.previous_resident_pages_per_layer < feedback_resident_floor;

    const bool below_overlap_target =
        params.previous_resident_pages_per_layer > target_resident_pages;
    const bool feedback_starved =
        params.previous_resident_pages_per_layer > feedback_resident_floor &&
        params.deadline_miss_ratio > MISS_THRESHOLD &&
        params.copy_engine_busy_ratio < COPY_SATURATED;
    const bool starved = below_overlap_target || feedback_starved;
    const bool overprovisioned = params.deadline_miss_ratio <= MISS_THRESHOLD &&
        params.copy_engine_busy_ratio < COPY_LIGHT &&
        params.ring_peak_occupancy_ratio < RING_LIGHT;

    result.starved_evaluations = starved ? params.starved_evaluations + 1 : 0;
    result.overprovisioned_evaluations = overprovisioned ?
        params.overprovisioned_evaluations + 1 : 0;

    const bool cooldown_complete = params.evaluations_since_repartition >=
        params.repartition_cooldown_evaluations;
    if (params.entering_decode_layout && below_overlap_target) {
        result.resident_pages_per_layer = target_resident_pages;
        result.ring_slots = params.total_pool_pages -
            result.resident_pages_per_layer*params.layer_count;
        result.starved_evaluations = 0;
        result.overprovisioned_evaluations = 0;
        result.changed = true;
    } else if (cooldown_complete && overgrown_ring) {
        result.resident_pages_per_layer = feedback_resident_floor;
        result.ring_slots = params.total_pool_pages -
            result.resident_pages_per_layer*params.layer_count;
        result.starved_evaluations = 0;
        result.overprovisioned_evaluations = 0;
        result.changed = true;
    } else if (cooldown_complete && starved &&
            result.starved_evaluations >= params.grow_hysteresis_evaluations &&
            result.resident_pages_per_layer > 0) {
        result.resident_pages_per_layer = below_overlap_target ?
            target_resident_pages : result.resident_pages_per_layer - 1;
        result.ring_slots = params.total_pool_pages -
            result.resident_pages_per_layer*params.layer_count;
        result.starved_evaluations = 0;
        result.overprovisioned_evaluations = 0;
        result.changed = true;
    } else if (cooldown_complete && overprovisioned &&
            result.overprovisioned_evaluations >= params.shrink_hysteresis_evaluations &&
            result.resident_pages_per_layer < target_resident_pages &&
            result.ring_slots >= params.minimum_ring_slots + params.layer_count) {
        ++result.resident_pages_per_layer;
        result.ring_slots -= params.layer_count;
        result.starved_evaluations = 0;
        result.overprovisioned_evaluations = 0;
        result.changed = true;
    }

    result.valid = true;
    return result;
}

llama_kv_stream_feedback_delta llama_kv_stream_feedback_delta_make(
        const llama_kv_stream_feedback_counters & current,
        const llama_kv_stream_feedback_counters & previous) {
    llama_kv_stream_feedback_delta result;
    if (current.deadline_samples < previous.deadline_samples ||
            current.deadline_misses < previous.deadline_misses) {
        result.error = "KV stream feedback counters moved backwards";
        return result;
    }

    result.deadline_samples = current.deadline_samples - previous.deadline_samples;
    result.deadline_misses = current.deadline_misses - previous.deadline_misses;
    if (result.deadline_misses > result.deadline_samples) {
        result.error = "KV stream deadline misses exceed samples";
        return result;
    }

    result.valid = true;
    result.has_evaluation = result.deadline_samples != 0;
    if (result.has_evaluation) {
        result.deadline_miss_ratio =
            double(result.deadline_misses)/double(result.deadline_samples);
    }
    return result;
}
