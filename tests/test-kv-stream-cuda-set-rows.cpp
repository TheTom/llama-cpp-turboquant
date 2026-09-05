#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "ggml.h"
#include "testing.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

constexpr int64_t CACHE_WIDTH = 256*4;
constexpr int64_t CACHE_ROWS = 512;
constexpr int64_t UPDATE_ROWS = 3;

std::vector<uint8_t> run_set_rows(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t cache_buft,
        ggml_type cache_type) {
    constexpr size_t N_TENSORS = 16;
    const size_t context_bytes = ggml_tensor_overhead()*N_TENSORS +
        ggml_graph_overhead_custom(N_TENSORS, false);
    const ggml_init_params params{context_bytes, nullptr, true};

    ggml_context_ptr cache_ctx(ggml_init(params));
    ggml_context_ptr compute_ctx(ggml_init(params));
    GGML_ASSERT(cache_ctx && compute_ctx);

    ggml_tensor * cache = ggml_new_tensor_2d(cache_ctx.get(), cache_type, CACHE_WIDTH, CACHE_ROWS);
    ggml_tensor * values = ggml_new_tensor_2d(compute_ctx.get(), GGML_TYPE_F32, CACHE_WIDTH, UPDATE_ROWS);
    ggml_tensor * indices = ggml_new_tensor_1d(compute_ctx.get(), GGML_TYPE_I32, UPDATE_ROWS);
    ggml_tensor * updated = ggml_set_rows(compute_ctx.get(), cache, values, indices);

    ggml_backend_buffer_ptr cache_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(cache_ctx.get(), cache_buft));
    ggml_backend_buffer_ptr compute_buffer(
        ggml_backend_alloc_ctx_tensors(compute_ctx.get(), backend));
    GGML_ASSERT(cache_buffer && compute_buffer);
    ggml_backend_buffer_clear(cache_buffer.get(), 0);

    std::vector<float> source(CACHE_WIDTH*UPDATE_ROWS);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = 0.5f*std::sin(float(i)*0.01953125f) - 0.25f*std::cos(float(i)*0.00390625f);
    }
    const std::vector<int32_t> rows{1, 257, 511};
    ggml_backend_tensor_set(values, source.data(), 0, source.size()*sizeof(float));
    ggml_backend_tensor_set(indices, rows.data(), 0, rows.size()*sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph_custom(compute_ctx.get(), N_TENSORS, false);
    ggml_build_forward_expand(graph, updated);
    GGML_ASSERT(ggml_backend_supports_op(backend, updated));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<uint8_t> result(ggml_nbytes(cache));
    ggml_backend_tensor_get(cache, result.data(), 0, result.size());
    return result;
}

} // namespace

int main() {
    testing t;

    t.test("KV stream quant types are classified", [](testing & t) {
        // TurboQuant: this loop asserts every quantized/f32/f16/bf16 type is
        // classified, including turbo2/3/4 (F16-conversion attention mode).
        // The fork's own weight-quant TQ/CR types are unrelated to KV cache
        // storage and are deliberately left unclassified (same as any other
        // type this doesn't recognize), gracefully falling through to
        // UNSUPPORTED in ggml_backend_cuda_kv_stream_get_attention_mode
        // rather than crashing.
        for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
            const ggml_type ggml_type_value = (ggml_type) type;
            if (!ggml_is_quantized(ggml_type_value) &&
                    ggml_type_value != GGML_TYPE_F32 &&
                    ggml_type_value != GGML_TYPE_F16 &&
                    ggml_type_value != GGML_TYPE_BF16) {
                continue;
            }
            switch (ggml_type_value) {
                case GGML_TYPE_TQ3_1S:
                case GGML_TYPE_TQ4_1S:
                case GGML_TYPE_Q5_CR:
                case GGML_TYPE_Q6_CR:
                case GGML_TYPE_Q8_CR:
                    continue;
                default:
                    break;
            }

            const auto capabilities =
                ggml_backend_cuda_kv_stream_get_type_capabilities(ggml_type_value);
            t.assert_true(ggml_type_name(ggml_type_value), capabilities.classified);
        }
    });

    t.test("Q8 K and Q4 V retain direct streamed attention", [](testing & t) {
        const auto k = ggml_backend_cuda_kv_stream_get_type_capabilities(GGML_TYPE_Q8_0);
        const auto v = ggml_backend_cuda_kv_stream_get_type_capabilities(GGML_TYPE_Q4_0);

        t.assert_true("Q8 supports direct attention", k.direct_attention);
        t.assert_true("Q4 supports direct attention", v.direct_attention);
        t.assert_equal(
            GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT,
            ggml_backend_cuda_kv_stream_get_attention_mode(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0));
    });

    t.test("all native CUDA flash-attention KV pairs use direct streaming", [](testing & t) {
        const ggml_type native_types[] = {
            GGML_TYPE_F16,
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0,
            GGML_TYPE_BF16,
        };

        for (const ggml_type type_k : native_types) {
            for (const ggml_type type_v : native_types) {
                t.assert_equal(
                    GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT,
                    ggml_backend_cuda_kv_stream_get_attention_mode(type_k, type_v));
            }
        }
    });

    t.test("all exposed KV-cache pairs select an optimized execution class", [](testing & t) {
        const ggml_type kv_types[] = {
            GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
            GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
        };
        size_t direct_pairs = 0;
        size_t fallback_pairs = 0;
        for (const ggml_type type_k : kv_types) {
            const auto k = ggml_backend_cuda_kv_stream_get_type_capabilities(type_k);
            for (const ggml_type type_v : kv_types) {
                const auto v = ggml_backend_cuda_kv_stream_get_type_capabilities(type_v);
                const auto expected = k.direct_attention && v.direct_attention ?
                    GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT :
                    GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16;
                const auto actual =
                    ggml_backend_cuda_kv_stream_get_attention_mode(type_k, type_v);
                if (!t.assert_equal(expected, actual)) {
                    std::fprintf(stderr, "mode mismatch K=%s V=%s\n",
                        ggml_type_name(type_k), ggml_type_name(type_v));
                    return;
                }
                direct_pairs += actual == GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT;
                fallback_pairs += actual == GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16;
            }
        }
        t.assert_equal(size_t(49), direct_pairs);
        t.assert_equal(size_t(32), fallback_pairs);
    });

    t.test("every exposed KV-cache type has a GPU online writer", [](testing & t) {
        const ggml_type kv_types[] = {
            GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
            GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
        };
        for (const ggml_type type : kv_types) {
            const auto capabilities =
                ggml_backend_cuda_kv_stream_get_type_capabilities(type);
            t.assert_true(ggml_type_name(type), capabilities.online_write);
            t.assert_true(ggml_type_name(type), capabilities.decode_f16);
        }
        const auto internal =
            ggml_backend_cuda_kv_stream_get_type_capabilities(GGML_TYPE_Q2_K);
        t.assert_true("internal Q2_K tensor format is classified", internal.classified);
        t.assert_true("internal Q2_K is not advertised as online writable", !internal.online_write);
        t.assert_equal(
            GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_UNSUPPORTED,
            ggml_backend_cuda_kv_stream_get_attention_mode(
                GGML_TYPE_Q2_K, GGML_TYPE_Q4_0));
    });

    t.test("CUDA backend reports executable KV stream pairs", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        using supported_fn_t = bool (*)(ggml_type, ggml_type);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto supported_fn = reinterpret_cast<supported_fn_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_type_pair_supported"));

        if (!t.assert_true("type-pair capability query is available", supported_fn != nullptr)) {
            return;
        }
        t.assert_true("Q8 K and Q4 V are executable",
            supported_fn(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0));
        t.assert_true("auxiliary Q8_1 pair is rejected",
            !supported_fn(GGML_TYPE_Q8_1, GGML_TYPE_Q8_1));
    });

    t.test("CUDA backend computes block-safe KV page geometry", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        using page_bytes_fn_t = bool (*)(
            ggml_type, ggml_type, uint32_t, uint32_t, uint32_t, uint32_t, size_t *);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto page_bytes_fn = reinterpret_cast<page_bytes_fn_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_page_bytes"));
        if (!t.assert_true("page geometry query is available", page_bytes_fn != nullptr)) {
            return;
        }

        size_t page_bytes = 0;
        t.assert_true("Q8/Q4 geometry is valid", page_bytes_fn(
            GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 256, 256, 4, 256, &page_bytes));
        t.assert_equal(
            256*(4*ggml_row_size(GGML_TYPE_Q8_0, 256) + 4*ggml_row_size(GGML_TYPE_Q4_0, 256)),
            page_bytes);

        size_t reversed_page_bytes = 0;
        t.assert_true("reversed geometry is valid", page_bytes_fn(
            GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, 256, 256, 4, 256, &reversed_page_bytes));
        t.assert_equal(page_bytes, reversed_page_bytes);

        t.assert_true("K quant rejects a partial superblock", !page_bytes_fn(
            GGML_TYPE_Q2_K, GGML_TYPE_Q4_0, 128, 256, 4, 256, &page_bytes));
        t.assert_true("K quant accepts a complete superblock", page_bytes_fn(
            GGML_TYPE_Q2_K, GGML_TYPE_Q4_0, 256, 256, 4, 256, &page_bytes));
        t.assert_true("overflow is rejected", !page_bytes_fn(
            GGML_TYPE_F32, GGML_TYPE_F32, UINT32_MAX, UINT32_MAX,
            UINT32_MAX, UINT32_MAX, &page_bytes));
    });

    t.test("conversion workspace is bounded to one page for every exposed KV pair", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        using workspace_fn_t = bool (*)(
            ggml_type, ggml_type, uint32_t, uint32_t, uint32_t, uint32_t, size_t *);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto workspace_fn = reinterpret_cast<workspace_fn_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_workspace_bytes"));
        if (!t.assert_true("workspace query is available", workspace_fn != nullptr)) {
            return;
        }

        constexpr uint32_t head_dim = 256;
        constexpr uint32_t head_count = 4;
        constexpr uint32_t page_tokens = 256;
        const size_t f16_k_page =
            ggml_row_size(GGML_TYPE_F16, head_dim)*head_count*page_tokens;
        const size_t f16_v_page = f16_k_page;
        const size_t expected_fallback = ((f16_k_page + 127) & ~size_t(127)) + f16_v_page;
        const ggml_type kv_types[] = {
            GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
            GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
        };

        for (const ggml_type type_k : kv_types) {
            for (const ggml_type type_v : kv_types) {
                size_t workspace_bytes = SIZE_MAX;
                if (!t.assert_true("exposed pair has a workspace policy", workspace_fn(
                        type_k, type_v, head_dim, head_dim, head_count,
                        page_tokens, &workspace_bytes))) {
                    std::fprintf(stderr, "workspace query failed K=%s V=%s\n",
                        ggml_type_name(type_k), ggml_type_name(type_v));
                    return;
                }
                const auto mode =
                    ggml_backend_cuda_kv_stream_get_attention_mode(type_k, type_v);
                const size_t expected = mode == GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_DIRECT ?
                    0 : expected_fallback;
                if (!t.assert_equal(expected, workspace_bytes)) {
                    std::fprintf(stderr, "workspace mismatch K=%s V=%s\n",
                        ggml_type_name(type_k), ggml_type_name(type_v));
                    return;
                }
            }
        }
    });

    t.test("mapped authoritative cache SET_ROWS matches CUDA quantization", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        ggml_backend_cuda_kv_stream_params params{};
        params.device = 0;
        params.stage_bytes = 512*1024;
        params.stage_slots = 1;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        const ggml_type kv_types[] = {
            GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
            GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
        };
        for (const ggml_type type : kv_types) {
            const std::vector<uint8_t> expected = run_set_rows(
                backend.get(), ggml_backend_get_default_buffer_type(backend.get()), type);
            const std::vector<uint8_t> actual = run_set_rows(
                backend.get(), ggml_backend_cuda_kv_stream_buffer_type(runtime), type);
            if (!t.assert_true(ggml_type_name(type), expected == actual)) {
                return;
            }
        }

        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    return t.summary();
}
