#include "llama-kv-stream-config.h"
#include "testing.h"

int main() {
    testing t;

    t.test("streaming is opt-in", [](testing & t) {
        llama_kv_stream_config config;
        const auto result = llama_kv_stream_config_validate(config);
        t.assert_true("disabled config is valid", result.valid);
        t.assert_true("disabled config remains disabled", !result.enabled);
    });

    t.test("supported target configuration is accepted", [](testing & t) {
        llama_kv_stream_config config;
        config.arena_bytes         = 64ULL*1024ULL*1024ULL;
        config.minimum_arena_bytes = 1664ULL*256ULL;
        config.unified_kv_cache    = true;
        config.type_pair_supported = true;
        config.context_default    = true;
        config.single_sequence    = true;
        config.flash_attention    = true;
        config.kv_offload         = true;

        const auto result = llama_kv_stream_config_validate(config);
        t.assert_true("config is valid", result.valid);
        t.assert_true("config is enabled", result.enabled);
    });

    t.test("each unsupported condition fails loudly", [](testing & t) {
        llama_kv_stream_config base;
        base.arena_bytes         = 64ULL*1024ULL*1024ULL;
        base.minimum_arena_bytes = 1664ULL*256ULL;
        base.unified_kv_cache    = true;
        base.type_pair_supported = true;
        base.context_default     = true;
        base.single_sequence     = true;
        base.flash_attention     = true;
        base.kv_offload          = true;

        auto expect_invalid = [&](const char * name, const llama_kv_stream_config & config) {
            const auto result = llama_kv_stream_config_validate(config);
            t.assert_true(name, !result.valid && !result.enabled && !result.error.empty());
        };

        auto config = base;
        config.unified_kv_cache = false;
        expect_invalid("non-unified KV cache", config);
        config = base;
        config.context_default = false;
        expect_invalid("draft/MTP context", config);
        config = base;
        config.single_sequence = false;
        expect_invalid("parallel sequences", config);
        config = base;
        config.flash_attention = false;
        expect_invalid("Flash Attention disabled", config);
        config = base;
        config.kv_offload = false;
        expect_invalid("KV offload disabled", config);
        config = base;
        config.type_pair_supported = false;
        expect_invalid("unsupported K/V type pair", config);
        config = base;
        config.arena_bytes = config.minimum_arena_bytes - 1;
        expect_invalid("arena below its minimum", config);
    });

    t.test("pool is partitioned evenly across layers with one scratch page", [](testing & t) {
        const auto layout = llama_kv_stream_pool_layout_make({
            /*.pool_bytes   =*/ 64ULL*1024ULL*1024ULL,
            /*.page_bytes   =*/ 1664ULL*256ULL,
            /*.layer_count  =*/ 16,
            /*.scratch_pages=*/ 1,
        });

        t.assert_true("layout is valid", layout.valid);
        t.assert_equal(uint32_t(9), layout.resident_pages_per_layer);
        t.assert_equal(uint32_t(9*256), layout.resident_tokens_per_layer);
        t.assert_equal(1664ULL*256ULL, layout.scratch_bytes);
        t.assert_equal(
            64ULL*1024ULL*1024ULL,
            layout.scratch_bytes + layout.resident_bytes + layout.unused_bytes);
    });

    t.test("pool rejects missing scratch or resident capacity", [](testing & t) {
        auto layout = llama_kv_stream_pool_layout_make({ 0, 1664ULL*256ULL, 16, 1 });
        t.assert_true("zero pool", !layout.valid);

        layout = llama_kv_stream_pool_layout_make({ 1664ULL*256ULL, 1664ULL*256ULL, 16, 1 });
        t.assert_true("scratch-only pool", !layout.valid);

        layout = llama_kv_stream_pool_layout_make({ 64ULL*1024ULL*1024ULL, 0, 16, 1 });
        t.assert_true("zero page", !layout.valid);
    });

    t.test("phase arena partitions KV below compute without overlap", [](testing & t) {
        const auto plan = llama_kv_stream_phase_plan_make({
            /*.arena_bytes        =*/ 4096ULL*1024ULL*1024ULL,
            /*.compute_bytes      =*/ 1192ULL*1024ULL*1024ULL,
            /*.compute_alignment  =*/ 256,
            /*.page_bytes         =*/ 1664ULL*256ULL,
            /*.conversion_bytes   =*/ 32ULL*1024ULL*1024ULL,
            /*.layer_count        =*/ 16,
            /*.minimum_ring_pages =*/ 8,
        });

        t.assert_true("plan is valid", plan.valid);
        t.assert_equal(uint64_t(0), plan.kv_offset);
        t.assert_equal(plan.kv_bytes, plan.compute_offset);
        t.assert_equal(4096ULL*1024ULL*1024ULL, plan.compute_offset + plan.compute_bytes);
        t.assert_equal(uint64_t(0), plan.compute_offset % 256);
        t.assert_equal(
            plan.kv_bytes,
            plan.resident_bytes + plan.ring_bytes + plan.conversion_bytes + plan.unused_bytes);
        t.assert_equal(
            uint64_t(plan.ring_pages)*1664ULL*256ULL,
            plan.ring_bytes);
    });

    t.test("decode reclaims the exact aligned prefill compute reduction", [](testing & t) {
        constexpr uint64_t MiB = 1024ULL*1024ULL;
        const auto prefill = llama_kv_stream_phase_plan_make({
            4096*MiB, 1192*MiB, MiB, 1664ULL*256ULL, 32*MiB, 16, 8,
        });
        const auto decode = llama_kv_stream_phase_plan_make({
            4096*MiB, 7*MiB, MiB, 1664ULL*256ULL, 32*MiB, 16, 8,
        });

        t.assert_true("prefill plan is valid", prefill.valid);
        t.assert_true("decode plan is valid", decode.valid);
        t.assert_equal(1185*MiB, decode.kv_bytes - prefill.kv_bytes);
        t.assert_equal(1185*MiB, prefill.compute_bytes - decode.compute_bytes);
    });

    t.test("prefill reserves a nonzero ring and one resident page per layer", [](testing & t) {
        constexpr uint64_t page_bytes = 1664ULL*256ULL;
        const auto plan = llama_kv_stream_phase_plan_make({
            256ULL*1024ULL*1024ULL,
            64ULL*1024ULL*1024ULL,
            256,
            page_bytes,
            8ULL*1024ULL*1024ULL,
            16,
            3,
        });

        t.assert_true("plan is valid", plan.valid);
        t.assert_equal(uint32_t(4), plan.ring_pages);
        t.assert_equal(4*page_bytes, plan.ring_bytes);
        t.assert_true("resident round retained", plan.resident_pages_per_layer >= 1);
        t.assert_equal(8ULL*1024ULL*1024ULL, plan.conversion_bytes);
    });

    t.test("phase arena supports page geometry from other KV quantizations", [](testing & t) {
        constexpr uint64_t page_bytes = 320ULL*1024ULL;
        const auto plan = llama_kv_stream_phase_plan_make({
            768ULL*1024ULL*1024ULL,
            128ULL*1024ULL*1024ULL,
            512,
            page_bytes,
            24ULL*1024ULL*1024ULL,
            24,
            11,
        });

        t.assert_true("plan is valid", plan.valid);
        t.assert_equal(uint32_t(27), plan.ring_pages);
        t.assert_equal(27*page_bytes, plan.ring_bytes);
        t.assert_equal(
            uint64_t(plan.resident_pages_per_layer)*page_bytes*24,
            plan.resident_bytes);
    });

    t.test("phase arena rejects insufficient and overflowing layouts", [](testing & t) {
        constexpr uint64_t page_bytes = 1664ULL*256ULL;
        auto plan = llama_kv_stream_phase_plan_make({
            64ULL*1024ULL*1024ULL,
            48ULL*1024ULL*1024ULL,
            256,
            page_bytes,
            8ULL*1024ULL*1024ULL,
            16,
            8,
        });
        t.assert_true("insufficient resident space", !plan.valid);

        plan = llama_kv_stream_phase_plan_make({
            UINT64_MAX,
            1,
            UINT64_MAX,
            page_bytes,
            0,
            16,
            1,
        });
        t.assert_true("alignment leaves no KV slice", !plan.valid);

        plan = llama_kv_stream_phase_plan_make({
            UINT64_MAX,
            1,
            1,
            UINT64_MAX,
            0,
            16,
            2,
        });
        t.assert_true("ring byte overflow", !plan.valid);
    });

    t.test("explicit decode phase overrides ambiguous batch shape", [](testing & t) {
        t.assert_true("automatic one-token batch is generation",
            llama_kv_stream_phase_is_generation(
                LLAMA_KV_STREAM_PHASE_AUTOMATIC, 1));
        t.assert_true("automatic multi-token batch is prompt",
            !llama_kv_stream_phase_is_generation(
                LLAMA_KV_STREAM_PHASE_AUTOMATIC, 2));
        t.assert_true("one-token prompt tail remains prompt",
            !llama_kv_stream_phase_is_generation(
                LLAMA_KV_STREAM_PHASE_PROMPT, 1));
        t.assert_true("explicit prompt chunk remains prompt",
            !llama_kv_stream_phase_is_generation(
                LLAMA_KV_STREAM_PHASE_PROMPT, 256));
        t.assert_true("explicit generation ignores token count",
            llama_kv_stream_phase_is_generation(
                LLAMA_KV_STREAM_PHASE_GENERATION, 4));
    });

    return t.summary();
}
