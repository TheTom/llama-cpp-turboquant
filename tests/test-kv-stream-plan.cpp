#include "llama-kv-stream-plan.h"
#include "testing.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr uint64_t MIB = 1024ULL*1024ULL;

constexpr uint32_t N_TARGET_LAYERS = 16;
constexpr uint32_t N_MTP_LAYERS    = 1;
constexpr uint32_t PAGE_TOKENS     = 256;
constexpr uint32_t STAGE_TOKENS    = 16*1024;
constexpr uint64_t BYTES_PER_TOKEN = 1664;

std::vector<llama_kv_stream_region> make_qwen38_regions(uint32_t n_kv) {
    llama_kv_stream_regions_params params;
    params.page_tokens = PAGE_TOKENS;

    for (uint32_t il = 0; il < N_TARGET_LAYERS; ++il) {
        llama_kv_stream_layer_layout layer;
        layer.role            = llama_kv_stream_region_role::target;
        layer.layer_id        = 3 + 4*il;
        layer.n_tokens        = n_kv;
        layer.bytes_per_token = BYTES_PER_TOKEN;
        layer.layer_priority  = il;
        layer.pin_tail        = true;
        params.layers.push_back(layer);
    }

    llama_kv_stream_layer_layout mtp;
    mtp.role            = llama_kv_stream_region_role::mtp;
    mtp.layer_id        = 64;
    mtp.n_tokens        = n_kv;
    mtp.bytes_per_token = BYTES_PER_TOKEN;
    mtp.pin_all         = true;
    params.layers.push_back(mtp);

    return llama_kv_stream_regions_make(params).regions;
}

llama_kv_stream_plan_params make_params(uint32_t n_kv, uint64_t pool_mib = 2526) {
    llama_kv_stream_plan_params params;
    params.pool_bytes       = pool_mib*MIB;
    params.stage_slot_bytes = STAGE_TOKENS*BYTES_PER_TOKEN;
    params.stage_slots      = 2;
    params.regions          = make_qwen38_regions(n_kv);
    return params;
}

size_t count_role(
        const llama_kv_stream_plan_params & params,
        const std::vector<size_t> & region_indices,
        llama_kv_stream_region_role role) {
    return std::count_if(region_indices.begin(), region_indices.end(), [&](size_t index) {
        return params.regions.at(index).role == role;
    });
}

} // namespace

int main() {
    testing t;

    t.test("small context remains fully resident while transfer slots stay reserved", [](testing & t) {
        const auto params = make_params(1024);
        const auto plan = llama_kv_stream_plan_make(params);

        t.assert_true("plan is valid", plan.valid);
        t.assert_equal(2*STAGE_TOKENS*BYTES_PER_TOKEN, plan.reserved_stage_bytes);
        t.assert_equal(uint64_t(0), plan.streamed_bytes);
        t.assert_equal(size_t(0), plan.streamed_regions.size());
        t.assert_equal(params.regions.size(), plan.resident_regions.size());
        t.assert_equal(
            plan.pool_bytes,
            plan.reserved_stage_bytes + plan.resident_bytes + plan.unused_bytes);
    });

    t.test("pinned MTP pages remain resident at 196K", [](testing & t) {
        const auto params = make_params(196608);
        const auto plan = llama_kv_stream_plan_make(params);

        t.assert_true("plan is valid", plan.valid);
        t.assert_equal(size_t(0), count_role(params, plan.streamed_regions, llama_kv_stream_region_role::mtp));
        t.assert_true(
            "some target pages stream after the pool fills",
            count_role(params, plan.streamed_regions, llama_kv_stream_region_role::target) > 0);
        t.assert_true(
            "resident bytes never exceed the pool",
            plan.reserved_stage_bytes + plan.resident_bytes <= plan.pool_bytes);
    });

    t.test("one padded context step increases streaming smoothly", [](testing & t) {
        const auto params_before = make_params(196608);
        const auto params_after  = make_params(196608 + PAGE_TOKENS);

        const auto before = llama_kv_stream_plan_make(params_before);
        const auto after  = llama_kv_stream_plan_make(params_after);

        t.assert_true("before plan is valid", before.valid);
        t.assert_true("after plan is valid", after.valid);
        t.assert_equal(
            uint64_t((N_TARGET_LAYERS + N_MTP_LAYERS)*PAGE_TOKENS)*BYTES_PER_TOKEN,
            after.streamed_bytes - before.streamed_bytes);
    });

    t.test("planning fails when pinned pages and stages do not fit", [](testing & t) {
        auto params = make_params(196608);
        params.pool_bytes = 300*MIB;

        const auto plan = llama_kv_stream_plan_make(params);
        t.assert_true("plan is rejected", !plan.valid);
        t.assert_true("error is reported", !plan.error.empty());
    });

    t.test("a streamed region must fit one transfer slot", [](testing & t) {
        llama_kv_stream_plan_params params;
        params.pool_bytes       = 1*MIB;
        params.stage_slot_bytes = 1*MIB;
        params.stage_slots      = 1;

        llama_kv_stream_region region;
        region.layer_id    = 3;
        region.token_count = 1;
        region.bytes       = 2*MIB;
        params.regions.push_back(region);

        const auto plan = llama_kv_stream_plan_make(params);
        t.assert_true("plan is rejected", !plan.valid);
        t.assert_true("error is reported", !plan.error.empty());
    });

    t.test("every region is assigned exactly once and accounting closes", [](testing & t) {
        for (uint32_t n_kv : { 1024U, 92160U, 196608U, 262144U }) {
            const auto params = make_params(n_kv);
            const auto plan = llama_kv_stream_plan_make(params);

            t.assert_true("plan is valid", plan.valid);

            std::vector<uint8_t> seen(params.regions.size(), 0);
            uint64_t logical_bytes = 0;

            for (const auto & region : params.regions) {
                logical_bytes += region.bytes;
            }

            bool indices_in_range = true;
            for (size_t index : plan.resident_regions) {
                if (index < seen.size()) {
                    ++seen[index];
                } else {
                    indices_in_range = false;
                }
            }

            for (size_t index : plan.streamed_regions) {
                if (index < seen.size()) {
                    ++seen[index];
                } else {
                    indices_in_range = false;
                }
            }

            t.assert_true("all assignment indices are in range", indices_in_range);
            t.assert_true("every region is assigned once", std::all_of(seen.begin(), seen.end(), [](uint8_t count) {
                return count == 1;
            }));
            t.assert_equal(logical_bytes, plan.resident_bytes + plan.streamed_bytes);
            t.assert_equal(
                plan.pool_bytes,
                plan.reserved_stage_bytes + plan.resident_bytes + plan.unused_bytes);
        }
    });

    t.test("streamed bytes grow monotonically without a context cliff", [](testing & t) {
        std::vector<uint32_t> contexts = { 1024, 65536, 89088 };
        for (uint32_t n_kv = 89344; n_kv <= 94208; n_kv += PAGE_TOKENS) {
            contexts.push_back(n_kv);
        }
        contexts.insert(contexts.end(), { 131072, 196608, 196864, 262144 });

        uint64_t previous_streamed = 0;
        uint32_t previous_context = 0;

        for (uint32_t n_kv : contexts) {
            const auto plan = llama_kv_stream_plan_make(make_params(n_kv));

            if (!t.assert_true("plan is valid", plan.valid)) {
                break;
            }

            t.assert_true("streamed bytes are monotonic", plan.streamed_bytes >= previous_streamed);
            if (previous_context != 0) {
                const uint64_t maximum_step =
                    uint64_t(N_TARGET_LAYERS + N_MTP_LAYERS)*(n_kv - previous_context)*BYTES_PER_TOKEN;
                t.assert_true(
                    "context growth cannot stream more bytes than the added logical KV",
                    plan.streamed_bytes - previous_streamed <= maximum_step);
            }

            previous_streamed = plan.streamed_bytes;
            previous_context  = n_kv;
        }
    });

    t.test("prefetch window prioritizes imminent pages and stops at lookahead", [](testing & t) {
        llama_kv_stream_prefetch_params params;
        params.current_attention = 4;
        params.lookahead_layers  = 3;
        params.stage_slot_bytes  = 4096;
        params.free_slots        = { 2, 0, 1, 3, 4, 5, 6, 7 };

        for (uint32_t attention = 4; attention <= 8; ++attention) {
            for (uint32_t page = 0; page < 2; ++page) {
                llama_kv_stream_prefetch_request request;
                request.layer_id        = int32_t(3 + 4*attention);
                request.attention_index = attention;
                request.page_index      = page;
                request.bytes           = 4096;
                params.requests.push_back(request);
            }
        }
        params.states.resize(params.requests.size(), llama_kv_stream_prefetch_state::pending);

        const auto dispatch = llama_kv_stream_prefetch_dispatch(params);
        t.assert_true("dispatch is valid", dispatch.valid);
        t.assert_equal(size_t(8), dispatch.assignments.size());
        t.assert_true("free slots are normalized", dispatch.assignments[0].slot == 0);
        t.assert_true("current attention is first", dispatch.assignments[0].request_index == 0);
        t.assert_true("x+3 is inside the window", dispatch.assignments.back().request_index == 7);
        t.assert_true("x+4 is outside the window", std::none_of(
            dispatch.assignments.begin(), dispatch.assignments.end(), [](const auto & assignment) {
                return assignment.request_index >= 8;
            }));
    });

    t.test("prefetch slot reuse waits for release and selects the earliest deadline", [](testing & t) {
        llama_kv_stream_prefetch_params params;
        params.current_attention = 0;
        params.lookahead_layers  = 3;
        params.stage_slot_bytes  = 4096;
        params.free_slots        = { 0, 1 };

        for (uint32_t attention = 0; attention < 4; ++attention) {
            llama_kv_stream_prefetch_request request;
            request.layer_id        = int32_t(3 + 4*attention);
            request.attention_index = attention;
            request.page_index      = 0;
            request.bytes           = 4096;
            params.requests.push_back(request);
        }
        params.states.resize(params.requests.size(), llama_kv_stream_prefetch_state::pending);

        auto dispatch = llama_kv_stream_prefetch_dispatch(params);
        t.assert_equal(size_t(2), dispatch.assignments.size());
        t.assert_equal(size_t(0), dispatch.assignments[0].request_index);
        t.assert_equal(size_t(1), dispatch.assignments[1].request_index);

        params.states[0] = llama_kv_stream_prefetch_state::consumed;
        params.states[1] = llama_kv_stream_prefetch_state::scheduled;
        params.free_slots = { dispatch.assignments[0].slot };
        dispatch = llama_kv_stream_prefetch_dispatch(params);
        t.assert_equal(size_t(1), dispatch.assignments.size());
        t.assert_equal(size_t(2), dispatch.assignments[0].request_index);
        t.assert_equal(uint32_t(0), dispatch.assignments[0].slot);
    });

    t.test("future mutable tails are not prefetched before their SET_ROWS producer", [](testing & t) {
        llama_kv_stream_prefetch_params params;
        params.current_attention = 2;
        params.lookahead_layers  = 3;
        params.stage_slot_bytes  = 4096;
        params.free_slots        = { 0, 1, 2 };

        llama_kv_stream_prefetch_request current_tail;
        current_tail.layer_id                 = 11;
        current_tail.attention_index          = 2;
        current_tail.page_index               = 9;
        current_tail.bytes                    = 4096;
        current_tail.producer_attention_index = 2;

        auto future_tail = current_tail;
        future_tail.layer_id                 = 15;
        future_tail.attention_index          = 3;
        future_tail.producer_attention_index = 3;

        auto future_stable = future_tail;
        future_stable.page_index               = 8;
        future_stable.producer_attention_index = -1;

        params.requests = { current_tail, future_tail, future_stable };
        params.states.resize(params.requests.size(), llama_kv_stream_prefetch_state::pending);

        auto dispatch = llama_kv_stream_prefetch_dispatch(params);
        t.assert_equal(size_t(1), dispatch.assignments.size());
        t.assert_equal(size_t(2), dispatch.assignments[0].request_index);

        params.current_producer_complete = true;
        dispatch = llama_kv_stream_prefetch_dispatch(params);
        t.assert_equal(size_t(2), dispatch.assignments.size());
        t.assert_equal(size_t(0), dispatch.assignments[0].request_index);
        t.assert_equal(size_t(2), dispatch.assignments[1].request_index);
    });

    t.test("adaptive prefetch considers every future attention deadline", [](testing & t) {
        llama_kv_stream_prefetch_params params;
        params.current_attention = 0;
        params.adaptive_lookahead = true;
        params.stage_slot_bytes  = 4096;
        params.free_slots        = { 0, 1, 2, 3 };

        for (uint32_t attention = 0; attention < N_TARGET_LAYERS; ++attention) {
            llama_kv_stream_prefetch_request request;
            request.layer_id        = int32_t(3 + 4*attention);
            request.attention_index = attention;
            request.page_index      = 7;
            request.bytes           = 4096;
            params.requests.push_back(request);
        }
        params.states.resize(params.requests.size(), llama_kv_stream_prefetch_state::pending);

        auto dispatch = llama_kv_stream_prefetch_dispatch(params);
        t.assert_true("dispatch is valid", dispatch.valid);
        t.assert_equal(size_t(4), dispatch.assignments.size());
        t.assert_equal(size_t(3), dispatch.assignments.back().request_index);

        params.states[0] = llama_kv_stream_prefetch_state::consumed;
        for (size_t i = 1; i < 4; ++i) {
            params.states[i] = llama_kv_stream_prefetch_state::scheduled;
        }
        params.free_slots = { 0 };
        dispatch = llama_kv_stream_prefetch_dispatch(params);
        t.assert_equal(size_t(1), dispatch.assignments.size());
        t.assert_equal(size_t(4), dispatch.assignments[0].request_index);
    });

    t.test("repeated deadline misses grow the ring by one balanced layer epoch", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                 = 160;
        params.layer_count                      = N_TARGET_LAYERS;
        params.active_pages_per_layer           = 12;
        params.minimum_ring_slots               = 16;
        params.previous_resident_pages_per_layer = 9;
        params.previous_ring_slots              = 16;
        params.deadline_miss_ratio               = 0.12;
        params.copy_engine_busy_ratio            = 0.70;
        params.starved_evaluations               = 2;
        params.grow_hysteresis_evaluations       = 3;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition changed", partition.changed);
        t.assert_equal(uint32_t(8), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(32), partition.ring_slots);
        t.assert_equal(uint32_t(0), partition.starved_evaluations);
    });

    t.test("feedback cannot demote beyond one epoch past the overlap target", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 7010;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 716;
        params.minimum_ring_slots                = 18;
        params.previous_resident_pages_per_layer = 416;
        params.previous_ring_slots               = 354;
        params.deadline_miss_ratio                = 0.10;
        params.copy_engine_busy_ratio             = 0.80;
        params.starved_evaluations                = 2;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition remains stable", !partition.changed);
        t.assert_equal(uint32_t(416), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(354), partition.ring_slots);
    });

    t.test("runaway feedback partition heals to the bounded floor", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 7010;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 716;
        params.minimum_ring_slots                = 18;
        params.previous_resident_pages_per_layer = 368;
        params.previous_ring_slots               = 1122;
        params.deadline_miss_ratio                = 0.10;
        params.copy_engine_busy_ratio             = 0.80;
        params.starved_evaluations                = 2;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition repairs the resident boundary", partition.changed);
        t.assert_equal(uint32_t(416), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(354), partition.ring_slots);
        t.assert_equal(uint32_t(0), partition.starved_evaluations);
    });

    t.test("PCIe saturation does not sacrifice more resident pages", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 160;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 12;
        params.minimum_ring_slots                = 16;
        params.previous_resident_pages_per_layer = 9;
        params.previous_ring_slots               = 16;
        params.deadline_miss_ratio                = 0.20;
        params.copy_engine_busy_ratio             = 0.99;
        params.starved_evaluations                = 10;
        params.grow_hysteresis_evaluations        = 3;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition remains stable", !partition.changed);
        t.assert_equal(uint32_t(9), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(16), partition.ring_slots);
    });

    t.test("undersized ring jumps to one-layer overlap target despite copy pressure", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 630;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 257;
        params.minimum_ring_slots                = 22;
        params.previous_resident_pages_per_layer = 38;
        params.previous_ring_slots               = 22;
        params.deadline_miss_ratio                = 0.40;
        params.copy_engine_busy_ratio             = 0.99;
        params.starved_evaluations                = 2;
        params.grow_hysteresis_evaluations        = 3;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition changed", partition.changed);
        t.assert_equal(uint32_t(23), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(262), partition.ring_slots);
    });

    t.test("decode transition selects its overlap target before feedback exists", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 6971;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 673;
        params.minimum_ring_slots                = 11;
        params.previous_resident_pages_per_layer = 435;
        params.previous_ring_slots               = 11;
        params.evaluations_since_repartition     = 0;
        params.entering_decode_layout            = true;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("decode transition changes immediately", partition.changed);
        t.assert_equal(uint32_t(418), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(283), partition.ring_slots);
    });

    t.test("copy pressure stops demotion after overlap target is reached", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 630;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 257;
        params.minimum_ring_slots                = 22;
        params.previous_resident_pages_per_layer = 23;
        params.previous_ring_slots               = 262;
        params.deadline_miss_ratio                = 0.20;
        params.copy_engine_busy_ratio             = 0.90;
        params.starved_evaluations                = 10;
        params.grow_hysteresis_evaluations        = 3;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition remains stable", !partition.changed);
        t.assert_equal(uint32_t(23), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(262), partition.ring_slots);
    });
    t.test("near-saturated copy traffic does not trigger a disruptive feedback epoch", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 6971;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 673;
        params.minimum_ring_slots                = 11;
        params.previous_resident_pages_per_layer = 418;
        params.previous_ring_slots               = 283;
        params.deadline_miss_ratio                = 0.053;
        params.copy_engine_busy_ratio             = 0.848;
        params.ring_peak_occupancy_ratio          = 1.0;
        params.starved_evaluations                = 2;
        params.evaluations_since_repartition      = 128;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("near-saturated partition remains stable", !partition.changed);
        t.assert_equal(uint32_t(418), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(283), partition.ring_slots);
        t.assert_equal(uint32_t(0), partition.starved_evaluations);
    });


    t.test("light utilization does not promote above overlap target", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 630;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 257;
        params.minimum_ring_slots                = 22;
        params.previous_resident_pages_per_layer = 23;
        params.previous_ring_slots               = 262;
        params.deadline_miss_ratio                = 0.0;
        params.copy_engine_busy_ratio             = 0.25;
        params.ring_peak_occupancy_ratio          = 0.20;
        params.overprovisioned_evaluations        = 7;
        params.shrink_hysteresis_evaluations      = 8;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition remains at target", !partition.changed);
        t.assert_equal(uint32_t(23), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(262), partition.ring_slots);
    });

    t.test("constrained pool demotes all resident pages when target is unreachable", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 39;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 65;
        params.minimum_ring_slots                = 23;
        params.previous_resident_pages_per_layer = 1;
        params.previous_ring_slots               = 23;
        params.deadline_miss_ratio                = 0.50;
        params.copy_engine_busy_ratio             = 0.99;
        params.starved_evaluations                = 2;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition changed", partition.changed);
        t.assert_equal(uint32_t(0), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(39), partition.ring_slots);
    });

    t.test("non-positive overlap target is rejected", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 160;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 12;
        params.minimum_ring_slots                = 16;
        params.previous_resident_pages_per_layer = 9;
        params.previous_ring_slots               = 16;
        params.target_ring_working_set_ratio      = 0.0;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is rejected", !partition.valid);
    });

    t.test("partition cooldown accumulates pressure without repeatedly resetting residency", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 160;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 12;
        params.minimum_ring_slots                = 16;
        params.previous_resident_pages_per_layer = 9;
        params.previous_ring_slots               = 16;
        params.deadline_miss_ratio               = 0.12;
        params.copy_engine_busy_ratio            = 0.70;
        params.starved_evaluations               = 2;
        params.evaluations_since_repartition     = 2;

        auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("cooldown preserves the resident boundary", !partition.changed);
        t.assert_equal(uint32_t(3), partition.starved_evaluations);

        params.evaluations_since_repartition = params.repartition_cooldown_evaluations;
        partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition changes after cooldown", partition.changed);
        t.assert_equal(uint32_t(8), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(32), partition.ring_slots);
    });

    t.test("sustained overprovision promotes one balanced resident epoch", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 160;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 12;
        params.minimum_ring_slots                = 16;
        params.previous_resident_pages_per_layer = 8;
        params.previous_ring_slots               = 32;
        params.deadline_miss_ratio                = 0.0;
        params.copy_engine_busy_ratio             = 0.25;
        params.ring_peak_occupancy_ratio          = 0.20;
        params.overprovisioned_evaluations        = 7;
        params.shrink_hysteresis_evaluations      = 8;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition changed", partition.changed);
        t.assert_equal(uint32_t(9), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(16), partition.ring_slots);
        t.assert_equal(uint32_t(0), partition.overprovisioned_evaluations);
    });

    t.test("cumulative CUDA feedback becomes one bounded evaluation delta", [](testing & t) {
        const auto delta = llama_kv_stream_feedback_delta_make(
            { 145, 17 }, { 120, 12 });
        t.assert_true("feedback delta is valid", delta.valid);
        t.assert_true("feedback contains a new evaluation", delta.has_evaluation);
        t.assert_equal(uint64_t(25), delta.deadline_samples);
        t.assert_equal(uint64_t(5), delta.deadline_misses);
        t.assert_true("deadline ratio is exact",
            std::abs(delta.deadline_miss_ratio - 0.20) < 1e-12);

        const auto unchanged = llama_kv_stream_feedback_delta_make(
            { 145, 17 }, { 145, 17 });
        t.assert_true("unchanged counters are valid", unchanged.valid);
        t.assert_true("unchanged counters do not invent an evaluation",
            !unchanged.has_evaluation);

        const auto reset = llama_kv_stream_feedback_delta_make(
            { 3, 1 }, { 145, 17 });
        t.assert_true("counter reset is rejected", !reset.valid);

        const auto impossible = llama_kv_stream_feedback_delta_make(
            { 150, 30 }, { 145, 17 });
        t.assert_true("more misses than samples are rejected", !impossible.valid);
    });

    t.test("duplicate logical regions are rejected", [](testing & t) {
        llama_kv_stream_plan_params params;
        params.pool_bytes       = 64*MIB;
        params.stage_slot_bytes = 1*MIB;
        params.stage_slots      = 1;

        llama_kv_stream_region region;
        region.layer_id    = 3;
        region.token_begin = 0;
        region.token_count = PAGE_TOKENS;
        region.bytes       = PAGE_TOKENS*BYTES_PER_TOKEN;

        params.regions.push_back(region);
        params.regions.push_back(region);

        const auto plan = llama_kv_stream_plan_make(params);
        t.assert_true("duplicate plan is rejected", !plan.valid);
    });

    t.test("overlapping regions in one logical layer are rejected", [](testing & t) {
        llama_kv_stream_plan_params params;
        params.pool_bytes       = 64*MIB;
        params.stage_slot_bytes = 1*MIB;
        params.stage_slots      = 1;

        llama_kv_stream_region first;
        first.layer_id    = 3;
        first.token_begin = 0;
        first.token_count = PAGE_TOKENS;
        first.bytes       = PAGE_TOKENS*BYTES_PER_TOKEN;

        auto second = first;
        second.token_begin = PAGE_TOKENS/2;

        params.regions.push_back(first);
        params.regions.push_back(second);

        const auto plan = llama_kv_stream_plan_make(params);
        t.assert_true("overlapping plan is rejected", !plan.valid);
    });

    t.test("overflowing token ranges are rejected", [](testing & t) {
        llama_kv_stream_plan_params params;
        params.pool_bytes       = 64*MIB;
        params.stage_slot_bytes = 1*MIB;
        params.stage_slots      = 1;

        llama_kv_stream_region region;
        region.layer_id    = 3;
        region.token_begin = std::numeric_limits<uint32_t>::max() - 127;
        region.token_count = PAGE_TOKENS;
        region.bytes       = PAGE_TOKENS*BYTES_PER_TOKEN;
        params.regions.push_back(region);

        const auto plan = llama_kv_stream_plan_make(params);
        t.assert_true("overflowing plan is rejected", !plan.valid);
    });

    t.test("extent includes pending speculative positions and page padding", [](testing & t) {
        llama_kv_stream_extent_params params;
        params.live_tokens    = 196607;
        params.reserve_tokens = 3;
        params.page_tokens    = PAGE_TOKENS;
        params.maximum_tokens = 262144;

        const auto extent = llama_kv_stream_extent_make(params);
        t.assert_true("extent is valid", extent.valid);
        t.assert_equal(uint32_t(196864), extent.tokens);
        t.assert_true("initial extent is growth", extent.grew);
        t.assert_true("initial extent is not shrink", !extent.shrunk);
    });

    t.test("growth is immediate", [](testing & t) {
        llama_kv_stream_extent_params params;
        params.live_tokens              = 131072;
        params.reserve_tokens           = 3;
        params.page_tokens              = PAGE_TOKENS;
        params.previous_extent          = 131072;
        params.shrink_hysteresis_tokens = 4096;
        params.maximum_tokens           = 262144;

        const auto extent = llama_kv_stream_extent_make(params);
        t.assert_true("extent is valid", extent.valid);
        t.assert_equal(uint32_t(131328), extent.tokens);
        t.assert_true("extent grew", extent.grew);
    });

    t.test("small rollback retains the previous extent", [](testing & t) {
        llama_kv_stream_extent_params params;
        params.live_tokens              = 196605;
        params.reserve_tokens           = 3;
        params.page_tokens              = PAGE_TOKENS;
        params.previous_extent          = 196864;
        params.shrink_hysteresis_tokens = 4096;
        params.maximum_tokens           = 262144;

        const auto extent = llama_kv_stream_extent_make(params);
        t.assert_true("extent is valid", extent.valid);
        t.assert_equal(uint32_t(196864), extent.tokens);
        t.assert_true("extent did not shrink", !extent.shrunk);
    });

    t.test("large shrink and forced reset release residency", [](testing & t) {
        llama_kv_stream_extent_params params;
        params.live_tokens              = 180000;
        params.page_tokens              = PAGE_TOKENS;
        params.previous_extent          = 196864;
        params.shrink_hysteresis_tokens = 4096;
        params.maximum_tokens           = 262144;

        auto extent = llama_kv_stream_extent_make(params);
        t.assert_true("large-shrink extent is valid", extent.valid);
        t.assert_equal(uint32_t(180224), extent.tokens);
        t.assert_true("large shrink is reported", extent.shrunk);

        params.live_tokens     = 0;
        params.force_shrink    = true;
        extent = llama_kv_stream_extent_make(params);
        t.assert_true("reset extent is valid", extent.valid);
        t.assert_equal(uint32_t(0), extent.tokens);
        t.assert_true("reset shrink is reported", extent.shrunk);
    });

    t.test("extent rejects invalid alignment and capacity overflow", [](testing & t) {
        llama_kv_stream_extent_params params;
        params.live_tokens    = 262143;
        params.reserve_tokens = 3;
        params.page_tokens    = PAGE_TOKENS;
        params.maximum_tokens = 262144;

        auto extent = llama_kv_stream_extent_make(params);
        t.assert_true("capacity overflow is rejected", !extent.valid);

        params.live_tokens    = 1024;
        params.reserve_tokens = 0;
        params.page_tokens    = 0;
        extent = llama_kv_stream_extent_make(params);
        t.assert_true("zero page size is rejected", !extent.valid);

        params.page_tokens      = PAGE_TOKENS;
        params.maximum_tokens   = 262143;
        extent = llama_kv_stream_extent_make(params);
        t.assert_true("unaligned maximum is rejected", !extent.valid);
    });

    t.test("region builder reproduces exact Qwen3.8 maximum-context geometry", [](testing & t) {
        llama_kv_stream_regions_params params;
        params.page_tokens = PAGE_TOKENS;

        for (uint32_t il = 0; il < N_TARGET_LAYERS; ++il) {
            llama_kv_stream_layer_layout layer;
            layer.role            = llama_kv_stream_region_role::target;
            layer.layer_id        = 3 + 4*il;
            layer.n_tokens        = 262144;
            layer.bytes_per_token = BYTES_PER_TOKEN;
            layer.layer_priority  = il;
            layer.pin_tail        = true;
            params.layers.push_back(layer);
        }

        llama_kv_stream_layer_layout mtp;
        mtp.role            = llama_kv_stream_region_role::mtp;
        mtp.layer_id        = 64;
        mtp.n_tokens        = 262144;
        mtp.bytes_per_token = BYTES_PER_TOKEN;
        mtp.pin_all         = true;
        params.layers.push_back(mtp);

        const auto built = llama_kv_stream_regions_make(params);
        t.assert_true("regions are valid", built.valid);
        t.assert_equal(size_t(17*1024), built.regions.size());
        t.assert_equal(uint64_t(7072)*MIB, built.total_bytes);

        const size_t pinned = std::count_if(built.regions.begin(), built.regions.end(), [](const auto & region) {
            return region.pinned;
        });
        t.assert_equal(size_t(1024 + N_TARGET_LAYERS), pinned);
        t.assert_true("every full page has exact bytes", std::all_of(
            built.regions.begin(), built.regions.end(), [](const auto & region) {
                return region.token_count == PAGE_TOKENS &&
                       region.bytes == PAGE_TOKENS*BYTES_PER_TOKEN;
            }));
    });

    t.test("region builder represents a partial tail without reserving a full page", [](testing & t) {
        llama_kv_stream_regions_params params;
        params.page_tokens = PAGE_TOKENS;

        llama_kv_stream_layer_layout layer;
        layer.layer_id        = 3;
        layer.n_tokens        = 196609;
        layer.bytes_per_token = BYTES_PER_TOKEN;
        layer.pin_tail        = true;
        params.layers.push_back(layer);

        const auto built = llama_kv_stream_regions_make(params);
        if (!t.assert_true("regions are valid", built.valid)) {
            return;
        }
        t.assert_equal(size_t(769), built.regions.size());

        if (built.regions.empty()) {
            return;
        }

        const auto & tail = built.regions.back();
        t.assert_equal(uint32_t(196608), tail.token_begin);
        t.assert_equal(uint32_t(1), tail.token_count);
        t.assert_equal(uint64_t(BYTES_PER_TOKEN), tail.bytes);
        t.assert_true("tail is pinned", tail.pinned);
    });

    t.test("region priority spreads the same context page across layers", [](testing & t) {
        llama_kv_stream_regions_params params;
        params.page_tokens = PAGE_TOKENS;

        for (uint32_t il = 0; il < 4; ++il) {
            llama_kv_stream_layer_layout layer;
            layer.layer_id        = 3 + 4*il;
            layer.n_tokens        = 2*PAGE_TOKENS;
            layer.bytes_per_token = BYTES_PER_TOKEN;
            layer.layer_priority  = il;
            params.layers.push_back(layer);
        }

        const auto built = llama_kv_stream_regions_make(params);
        t.assert_true("regions are valid", built.valid);

        uint64_t maximum_page_zero_priority = 0;
        uint64_t minimum_page_one_priority = std::numeric_limits<uint64_t>::max();
        for (const auto & region : built.regions) {
            if (region.token_begin == 0) {
                maximum_page_zero_priority = std::max(maximum_page_zero_priority, region.residency_priority);
            } else {
                minimum_page_one_priority = std::min(minimum_page_one_priority, region.residency_priority);
            }
        }

        t.assert_true(
            "all layers of one context page are preferred before the next page",
            maximum_page_zero_priority < minimum_page_one_priority);
    });

    t.test("region builder rejects invalid and overflowing layouts", [](testing & t) {
        llama_kv_stream_regions_params params;
        params.page_tokens = 0;
        auto built = llama_kv_stream_regions_make(params);
        t.assert_true("zero page size is rejected", !built.valid);

        params.page_tokens = PAGE_TOKENS;
        llama_kv_stream_layer_layout layer;
        layer.layer_id        = 3;
        layer.n_tokens        = 2;
        layer.bytes_per_token = std::numeric_limits<uint64_t>::max();
        params.layers.push_back(layer);
        built = llama_kv_stream_regions_make(params);
        t.assert_true("byte overflow is rejected", !built.valid);

        params.layers.clear();
        layer.n_tokens        = PAGE_TOKENS;
        layer.bytes_per_token = BYTES_PER_TOKEN;
        params.layers.push_back(layer);
        params.layers.push_back(layer);
        built = llama_kv_stream_regions_make(params);
        t.assert_true("duplicate logical layers are rejected", !built.valid);
    });

    return t.summary();
}
