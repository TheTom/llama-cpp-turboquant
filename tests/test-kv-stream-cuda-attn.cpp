#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-cuda/kv-stream-span-tuner.h"
#include "ggml.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

namespace {

constexpr int64_t HEAD_DIM = 256;
constexpr int64_t N_KV_HEAD = 4;
constexpr int64_t N_Q_HEAD  = 24;

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1)/alignment*alignment;
}

struct attention_inputs {
    ggml_type type_k = GGML_TYPE_Q8_0;
    ggml_type type_v = GGML_TYPE_Q4_0;
    std::vector<float> q;
    std::vector<uint8_t> k;
    std::vector<uint8_t> v;
    std::vector<uint16_t> mask;
};

attention_inputs make_inputs(
        int64_t n_kv, int64_t n_batch, int64_t query_start = 0,
        ggml_type type_k = GGML_TYPE_Q8_0, ggml_type type_v = GGML_TYPE_Q4_0) {
    attention_inputs result;
    result.type_k = type_k;
    result.type_v = type_v;

    result.q.resize(HEAD_DIM*n_batch*N_Q_HEAD);
    for (size_t i = 0; i < result.q.size(); ++i) {
        result.q[i] = 0.15f*std::sin(float(i)*0.03125f) + 0.05f*std::cos(float(i)*0.0078125f);
    }

    const int64_t nrows = n_kv*N_KV_HEAD;
    std::vector<float> source(HEAD_DIM*nrows);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = 0.4f*std::sin(float(i)*0.001953125f) + 0.2f*std::cos(float(i)*0.00048828125f);
    }

    result.k.resize(ggml_row_size(type_k, HEAD_DIM)*nrows);
    const size_t k_written = ggml_quantize_chunk(
        type_k, source.data(), result.k.data(), 0, nrows, HEAD_DIM, nullptr);
    GGML_ASSERT(k_written == result.k.size());

    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = 0.35f*std::cos(float(i)*0.00146484375f) - 0.1f*std::sin(float(i)*0.00390625f);
    }
    result.v.resize(ggml_row_size(type_v, HEAD_DIM)*nrows);
    const size_t v_written = ggml_quantize_chunk(
        type_v, source.data(), result.v.data(), 0, nrows, HEAD_DIM, nullptr);
    GGML_ASSERT(v_written == result.v.size());

    result.mask.resize(n_kv*n_batch);
    for (int64_t batch = 0; batch < n_batch; ++batch) {
        for (int64_t token = 0; token < n_kv; ++token) {
            // Vary both token blocks and both query rows. This catches a
            // streamed implementation that offsets the first mask row but
            // accidentally uses the compact block width as the next-row pitch.
            const float bias = token <= query_start + batch ?
                -0.015625f*float((token + 73*batch) % 127) : -INFINITY;
            result.mask[batch*n_kv + token] = ggml_fp32_to_fp16(bias);
        }
    }
    return result;
}

attention_inputs make_f16_reference(const attention_inputs & inputs, int64_t n_kv) {
    attention_inputs result;
    result.type_k = GGML_TYPE_F16;
    result.type_v = GGML_TYPE_F16;
    result.q = inputs.q;
    result.mask = inputs.mask;

    const int64_t nrows = n_kv*N_KV_HEAD;
    result.k.resize(ggml_row_size(GGML_TYPE_F16, HEAD_DIM)*nrows);
    result.v.resize(ggml_row_size(GGML_TYPE_F16, HEAD_DIM)*nrows);
    std::vector<float> row(HEAD_DIM);

    auto convert = [&](ggml_type type, const std::vector<uint8_t> & source,
            std::vector<uint8_t> & destination) {
        const size_t source_row_bytes = ggml_row_size(type, HEAD_DIM);
        const size_t destination_row_bytes = ggml_row_size(GGML_TYPE_F16, HEAD_DIM);
        const auto * traits = ggml_get_type_traits(type);
        for (int64_t i = 0; i < nrows; ++i) {
            const void * source_row = source.data() + size_t(i)*source_row_bytes;
            if (type == GGML_TYPE_F32) {
                std::memcpy(row.data(), source_row, HEAD_DIM*sizeof(float));
            } else {
                GGML_ASSERT(traits->to_float != nullptr);
                traits->to_float(source_row, row.data(), HEAD_DIM);
            }
            ggml_fp32_to_fp16_row(
                row.data(),
                reinterpret_cast<ggml_fp16_t *>(
                    destination.data() + size_t(i)*destination_row_bytes),
                HEAD_DIM);
        }
    };

    convert(inputs.type_k, inputs.k, result.k);
    convert(inputs.type_v, inputs.v, result.v);
    return result;
}

std::vector<float> run_attention(
        ggml_backend_t backend,
        const attention_inputs & inputs,
        ggml_backend_buffer_type_t kv_buft,
        int64_t n_kv,
        int64_t n_batch,
        int repeats = 1,
        int64_t update_rows = 1,
        bool change_updates = false,
        ggml_type index_type = GGML_TYPE_I32,
        bool index_on_host = true,
        ggml_backend_cuda_kv_stream_runtime_t dirty_runtime = nullptr,
        bool change_indices = false,
        bool replace_cache = false,
        uint64_t graph_uid = 0) {
    constexpr size_t N_TENSORS = 32;
    const size_t context_bytes = ggml_tensor_overhead()*N_TENSORS + ggml_graph_overhead_custom(N_TENSORS, false);

    ggml_init_params params{
        /* .mem_size   = */ context_bytes,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr compute_ctx(ggml_init(params));
    ggml_context_ptr kv_ctx(ggml_init(params));
    ggml_context_ptr index_ctx(ggml_init(params));
    GGML_ASSERT(compute_ctx && kv_ctx && index_ctx);

    ggml_tensor * q = ggml_new_tensor_4d(
        compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM, n_batch, N_Q_HEAD, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(
        compute_ctx.get(), GGML_TYPE_F16, n_kv, n_batch, 1, 1);
    ggml_tensor * k_storage = ggml_new_tensor_2d(
        kv_ctx.get(), inputs.type_k, HEAD_DIM*N_KV_HEAD, n_kv);
    ggml_tensor * v_storage = ggml_new_tensor_2d(
        kv_ctx.get(), inputs.type_v, HEAD_DIM*N_KV_HEAD, n_kv);
    ggml_tensor * k_cache = ggml_view_4d(
        kv_ctx.get(), k_storage, HEAD_DIM, N_KV_HEAD, n_kv, 1,
        ggml_row_size(inputs.type_k, HEAD_DIM),
        ggml_row_size(inputs.type_k, HEAD_DIM*N_KV_HEAD),
        ggml_row_size(inputs.type_k, HEAD_DIM*N_KV_HEAD)*n_kv, 0);
    ggml_tensor * v_cache = ggml_view_4d(
        kv_ctx.get(), v_storage, HEAD_DIM, N_KV_HEAD, n_kv, 1,
        ggml_row_size(inputs.type_v, HEAD_DIM),
        ggml_row_size(inputs.type_v, HEAD_DIM*N_KV_HEAD),
        ggml_row_size(inputs.type_v, HEAD_DIM*N_KV_HEAD)*n_kv, 0);
    ggml_tensor * k = ggml_permute(kv_ctx.get(), k_cache, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(kv_ctx.get(), v_cache, 0, 2, 1, 3);

    ggml_tensor * k_update = ggml_new_tensor_2d(
        compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM*N_KV_HEAD, update_rows);
    ggml_tensor * v_update = ggml_new_tensor_2d(
        compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM*N_KV_HEAD, update_rows);
    ggml_tensor * update_index = ggml_new_tensor_1d(index_ctx.get(), index_type, update_rows);
    ggml_tensor * updated_k = ggml_set_rows(compute_ctx.get(), k_storage, k_update, update_index);
    ggml_tensor * updated_v = ggml_set_rows(compute_ctx.get(), v_storage, v_update, update_index);

    ggml_tensor * out = ggml_flash_attn_ext(
        compute_ctx.get(), q, k, v, mask, 1.0f/std::sqrt(float(HEAD_DIM)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_set_name(out, "streamed-attention-output");

    ggml_backend_buffer_ptr kv_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(kv_ctx.get(), kv_buft));
    ggml_backend_buffer_ptr compute_buffer(
        ggml_backend_alloc_ctx_tensors(compute_ctx.get(), backend));
    ggml_backend_buffer_ptr index_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(index_ctx.get(), index_on_host ?
            ggml_backend_cuda_host_buffer_type() : ggml_backend_get_default_buffer_type(backend)));
    GGML_ASSERT(kv_buffer && compute_buffer && index_buffer);

    ggml_backend_tensor_set(q, inputs.q.data(), 0, inputs.q.size()*sizeof(float));
    ggml_backend_tensor_set(k_storage, inputs.k.data(), 0, inputs.k.size());
    ggml_backend_tensor_set(v_storage, inputs.v.data(), 0, inputs.v.size());
    ggml_backend_tensor_set(mask, inputs.mask.data(), 0, inputs.mask.size()*sizeof(uint16_t));

    std::vector<float> k_update_data(HEAD_DIM*N_KV_HEAD*update_rows);
    std::vector<float> v_update_data(HEAD_DIM*N_KV_HEAD*update_rows);
    for (size_t i = 0; i < k_update_data.size(); ++i) {
        k_update_data[i] = 0.6f*std::sin(float(i)*0.0234375f);
        v_update_data[i] = 0.4f*std::cos(float(i)*0.017578125f);
    }
    ggml_backend_tensor_set(k_update, k_update_data.data(), 0, k_update_data.size()*sizeof(float));
    ggml_backend_tensor_set(v_update, v_update_data.data(), 0, v_update_data.size()*sizeof(float));
    std::vector<int64_t> dirty_rows(update_rows);
    for (int64_t row = 0; row < update_rows; ++row) { dirty_rows[row] = row; }
    if (index_type == GGML_TYPE_I32) {
        std::vector<int32_t> update_index_data(update_rows);
        for (int64_t row = 0; row < update_rows; ++row) { update_index_data[row] = int32_t(row); }
        ggml_backend_tensor_set(
            update_index, update_index_data.data(), 0, update_index_data.size()*sizeof(int32_t));
    } else {
        GGML_ASSERT(index_type == GGML_TYPE_I64);
        ggml_backend_tensor_set(
            update_index, dirty_rows.data(), 0, dirty_rows.size()*sizeof(int64_t));
    }

    ggml_cgraph * graph = ggml_new_graph_custom(compute_ctx.get(), N_TENSORS, false);
    graph->uid = graph_uid;
    ggml_build_forward_expand(graph, updated_k);
    ggml_build_forward_expand(graph, updated_v);
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_supports_op(backend, updated_k));
    GGML_ASSERT(ggml_backend_supports_op(backend, updated_v));
    if (!ggml_backend_supports_op(backend, out)) {
        std::fprintf(stderr, "unsupported attention K=%s V=%s buft=%s\n",
            ggml_type_name(inputs.type_k), ggml_type_name(inputs.type_v),
            ggml_backend_buft_name(kv_buft));
    }
    GGML_ASSERT(ggml_backend_supports_op(backend, out));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if (replace_cache && repeat == 3) {
            std::vector<uint8_t> zero_k(inputs.k.size(), 0);
            std::vector<uint8_t> zero_v(inputs.v.size(), 0);
            ggml_backend_tensor_set(k_storage, zero_k.data(), 0, zero_k.size());
            ggml_backend_tensor_set(v_storage, zero_v.data(), 0, zero_v.size());
        }
        if (change_indices) {
            GGML_ASSERT(index_type == GGML_TYPE_I64);
            for (int64_t row = 0; row < update_rows; ++row) {
                dirty_rows[row] = (int64_t(repeat)*update_rows + row)%n_kv;
            }
            ggml_backend_tensor_set(
                update_index, dirty_rows.data(), 0, dirty_rows.size()*sizeof(int64_t));
        }
        if (dirty_runtime != nullptr) {
            GGML_ASSERT(ggml_backend_cuda_kv_stream_mark_dirty_rows(
                dirty_runtime, dirty_rows.data(), dirty_rows.size()));
        }
        if (repeat > 0 && change_updates) {
            for (float & value : k_update_data) { value = -2.0f*value; }
            for (float & value : v_update_data) { value = -2.0f*value; }
            ggml_backend_tensor_set(k_update, k_update_data.data(), 0, k_update_data.size()*sizeof(float));
            ggml_backend_tensor_set(v_update, v_update_data.data(), 0, v_update_data.size()*sizeof(float));
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    }

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size()*sizeof(float));
    return result;
}

std::vector<float> run_attention_layers(
        ggml_backend_t backend,
        const std::vector<attention_inputs> & layers,
        ggml_backend_buffer_type_t kv_buft,
        int64_t n_kv,
        int64_t n_batch,
        int repeats = 1,
        int64_t update_rows = 1,
        ggml_type index_type = GGML_TYPE_I32,
        ggml_backend_cuda_kv_stream_runtime_t dirty_runtime = nullptr,
        uint32_t layout_after_first = 0) {
    constexpr size_t N_TENSORS = 256;
    const size_t context_bytes = ggml_tensor_overhead()*N_TENSORS +
        ggml_graph_overhead_custom(N_TENSORS, false);

    ggml_init_params params{
        /* .mem_size   = */ context_bytes,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr compute_ctx(ggml_init(params));
    ggml_context_ptr kv_ctx(ggml_init(params));
    GGML_ASSERT(compute_ctx && kv_ctx);

    struct layer_tensors {
        ggml_tensor * q;
        ggml_tensor * mask;
        ggml_tensor * k_storage;
        ggml_tensor * v_storage;
        ggml_tensor * k_update;
        ggml_tensor * v_update;
        ggml_tensor * update_index;
        ggml_tensor * updated_k;
        ggml_tensor * updated_v;
        ggml_tensor * out;
    };
    std::vector<layer_tensors> tensors;
    tensors.reserve(layers.size());

    for (size_t layer = 0; layer < layers.size(); ++layer) {
        layer_tensors current{};
        current.q = ggml_new_tensor_4d(
            compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM, n_batch, N_Q_HEAD, 1);
        current.mask = ggml_new_tensor_4d(
            compute_ctx.get(), GGML_TYPE_F16, n_kv, n_batch, 1, 1);
        current.k_storage = ggml_new_tensor_2d(
            kv_ctx.get(), GGML_TYPE_Q8_0, HEAD_DIM*N_KV_HEAD, n_kv);
        current.v_storage = ggml_new_tensor_2d(
            kv_ctx.get(), GGML_TYPE_Q4_0, HEAD_DIM*N_KV_HEAD, n_kv);
        ggml_tensor * k_cache = ggml_view_4d(
            kv_ctx.get(), current.k_storage, HEAD_DIM, N_KV_HEAD, n_kv, 1,
            ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM),
            ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM*N_KV_HEAD),
            ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM*N_KV_HEAD)*n_kv, 0);
        ggml_tensor * v_cache = ggml_view_4d(
            kv_ctx.get(), current.v_storage, HEAD_DIM, N_KV_HEAD, n_kv, 1,
            ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM),
            ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM*N_KV_HEAD),
            ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM*N_KV_HEAD)*n_kv, 0);
        ggml_tensor * k = ggml_permute(kv_ctx.get(), k_cache, 0, 2, 1, 3);
        ggml_tensor * v = ggml_permute(kv_ctx.get(), v_cache, 0, 2, 1, 3);

        current.k_update = ggml_new_tensor_2d(
            compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM*N_KV_HEAD, update_rows);
        current.v_update = ggml_new_tensor_2d(
            compute_ctx.get(), GGML_TYPE_F32, HEAD_DIM*N_KV_HEAD, update_rows);
        current.update_index = ggml_new_tensor_1d(compute_ctx.get(), index_type, update_rows);
        current.updated_k = ggml_set_rows(
            compute_ctx.get(), current.k_storage, current.k_update, current.update_index);
        current.updated_v = ggml_set_rows(
            compute_ctx.get(), current.v_storage, current.v_update, current.update_index);
        current.out = ggml_flash_attn_ext(
            compute_ctx.get(), current.q, k, v, current.mask,
            1.0f/std::sqrt(float(HEAD_DIM)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(current.out, GGML_PREC_F32);
        tensors.push_back(current);
    }

    ggml_backend_buffer_ptr kv_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(kv_ctx.get(), kv_buft));
    ggml_backend_buffer_ptr compute_buffer(
        ggml_backend_alloc_ctx_tensors(compute_ctx.get(), backend));
    GGML_ASSERT(kv_buffer && compute_buffer);

    std::vector<int64_t> dirty_rows(update_rows);
    for (int64_t row = 0; row < update_rows; ++row) { dirty_rows[row] = row; }
    for (size_t layer = 0; layer < layers.size(); ++layer) {
        const auto & input = layers[layer];
        auto & current = tensors[layer];
        ggml_backend_tensor_set(current.q, input.q.data(), 0, input.q.size()*sizeof(float));
        ggml_backend_tensor_set(current.k_storage, input.k.data(), 0, input.k.size());
        ggml_backend_tensor_set(current.v_storage, input.v.data(), 0, input.v.size());
        ggml_backend_tensor_set(current.mask, input.mask.data(), 0, input.mask.size()*sizeof(uint16_t));

        std::vector<float> k_update_data(HEAD_DIM*N_KV_HEAD*update_rows);
        std::vector<float> v_update_data(HEAD_DIM*N_KV_HEAD*update_rows);
        for (size_t i = 0; i < k_update_data.size(); ++i) {
            k_update_data[i] = (0.5f + 0.03f*layer)*std::sin(float(i)*0.0234375f);
            v_update_data[i] = (0.3f + 0.02f*layer)*std::cos(float(i)*0.017578125f);
        }
        ggml_backend_tensor_set(current.k_update, k_update_data.data(), 0,
            k_update_data.size()*sizeof(float));
        ggml_backend_tensor_set(current.v_update, v_update_data.data(), 0,
            v_update_data.size()*sizeof(float));
        if (index_type == GGML_TYPE_I32) {
            std::vector<int32_t> rows_i32(update_rows);
            for (int64_t row = 0; row < update_rows; ++row) { rows_i32[row] = int32_t(row); }
            ggml_backend_tensor_set(
                current.update_index, rows_i32.data(), 0, rows_i32.size()*sizeof(int32_t));
        } else {
            GGML_ASSERT(index_type == GGML_TYPE_I64);
            ggml_backend_tensor_set(
                current.update_index, dirty_rows.data(), 0, dirty_rows.size()*sizeof(int64_t));
        }
    }

    ggml_cgraph * graph = ggml_new_graph_custom(compute_ctx.get(), N_TENSORS, false);
    for (auto & current : tensors) {
        ggml_build_forward_expand(graph, current.updated_k);
        ggml_build_forward_expand(graph, current.updated_v);
        ggml_build_forward_expand(graph, current.out);
    }
    for (int repeat = 0; repeat < repeats; ++repeat) {
        if (dirty_runtime != nullptr) {
            GGML_ASSERT(ggml_backend_cuda_kv_stream_mark_dirty_rows(
                dirty_runtime, dirty_rows.data(), dirty_rows.size()));
        }
        GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        if (repeat == 0 && layout_after_first != 0) {
            GGML_ASSERT(ggml_backend_cuda_kv_stream_set_decode_layout(
                dirty_runtime, layout_after_first));
        }
    }

    std::vector<float> result;
    for (auto & current : tensors) {
        const size_t begin = result.size();
        result.resize(begin + ggml_nelements(current.out));
        ggml_backend_tensor_get(current.out, result.data() + begin, 0,
            ggml_nbytes(current.out));
    }
    return result;
}

} // namespace

int main() {
    testing t;

    t.test("decode span tuner selects the faster measured mode per layout", [](testing & t) {
        ggml_cuda_kv_stream_span_tuner production_tuner;
        production_tuner.observe(100.0, /* streamed = */ true, /* bounded = */ false);
        for (uint32_t sample = 0; sample < 4; ++sample) {
            production_tuner.observe(10.0, /* streamed = */ true, /* bounded = */ false);
        }
        t.assert_true("production tuner does not decide from four samples",
            !production_tuner.use_bounded() && !production_tuner.selected());

        ggml_cuda_kv_stream_span_tuner tuner(/* trial_samples = */ 2, 0.005, /* warmup_samples = */ 1);

        tuner.observe(100.0, /* streamed = */ false, /* bounded = */ false);
        t.assert_true("non-streamed graphs do not start a trial", !tuner.use_bounded());
        t.assert_true("non-streamed graphs do not select a mode", !tuner.selected());

        tuner.observe(100.0, /* streamed = */ true, /* bounded = */ false);
        t.assert_true("unbounded warmup is not measured", !tuner.use_bounded());

        tuner.observe(10.0, /* streamed = */ true, /* bounded = */ false);
        tuner.observe(10.2, /* streamed = */ true, /* bounded = */ false);
        t.assert_true("tuner advances to bounded trials", tuner.use_bounded());
        t.assert_true("both modes are measured before selection", !tuner.selected());

        tuner.observe(100.0, /* streamed = */ true, /* bounded = */ true);
        t.assert_true("bounded warmup is not measured", !tuner.selected());

        tuner.observe(8.0, /* streamed = */ true, /* bounded = */ true);
        tuner.observe(8.2, /* streamed = */ true, /* bounded = */ true);
        t.assert_true("bounded mode is selected when materially faster", tuner.selected());
        t.assert_true("bounded mode remains active after selection", tuner.use_bounded());

        tuner.reset();
        tuner.observe(100.0, /* streamed = */ true, /* bounded = */ false);
        tuner.observe(10.0, /* streamed = */ true, /* bounded = */ false);
        tuner.observe(10.0, /* streamed = */ true, /* bounded = */ false);
        tuner.observe(100.0, /* streamed = */ true, /* bounded = */ true);
        tuner.observe(10.0, /* streamed = */ true, /* bounded = */ true);
        tuner.observe(10.0, /* streamed = */ true, /* bounded = */ true);
        t.assert_true("tuner selects after both trials", tuner.selected());
        t.assert_true("noise does not displace the ordinary kernel", !tuner.use_bounded());

        tuner.observe(1.0, /* streamed = */ true, /* bounded = */ true);
        t.assert_true("selection remains stable until layout reset", !tuner.use_bounded());
    });

    t.test("all native CUDA KV pairs preserve streamed prefill results", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 4;
        const ggml_type native_types[] = {
            GGML_TYPE_F16,
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0,
            GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0,
            GGML_TYPE_BF16,
        };

        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        for (const ggml_type type_k : native_types) {
            for (const ggml_type type_v : native_types) {
                const attention_inputs inputs =
                    make_inputs(n_kv, n_batch, n_kv - n_batch, type_k, type_v);
                const std::vector<float> expected = run_attention(
                    backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
                    n_kv, n_batch);

                const size_t k_page_bytes =
                    ggml_row_size(type_k, HEAD_DIM)*N_KV_HEAD*256;
                const size_t v_page_bytes =
                    ggml_row_size(type_v, HEAD_DIM)*N_KV_HEAD*256;
                const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
                ggml_backend_cuda_kv_stream_params params{};
                params.device      = 0;
                params.stage_bytes = page_bytes;
                params.stage_slots = 1;
                auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
                if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
                    return;
                }

                const std::vector<float> actual = run_attention(
                    backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
                    n_kv, n_batch);
                const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
                ggml_backend_cuda_kv_stream_runtime_free(runtime);

                if (!t.assert_equal(expected.size(), actual.size())) {
                    return;
                }
                float max_abs = 0.0f;
                for (size_t i = 0; i < expected.size(); ++i) {
                    max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
                }
                if (!std::isfinite(max_abs) || max_abs > 5e-4f ||
                        stats.asynchronous_page_uploads == 0) {
                    std::fprintf(stderr,
                        "native pair K=%s V=%s max_abs=%g async_uploads=%llu\n",
                        ggml_type_name(type_k), ggml_type_name(type_v), max_abs,
                        (unsigned long long) stats.asynchronous_page_uploads);
                }
                t.assert_true("native pair executes streamed attention",
                    stats.asynchronous_page_uploads > 0);
                t.assert_true("native pair remains numerically equivalent",
                    std::isfinite(max_abs) && max_abs <= 5e-4f);
            }
        }
    });

    t.test("all bounded-fallback KV pairs preserve streamed prefill results", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 4;
        const ggml_type kv_types[] = {
            GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
            GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
        };

        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        for (const ggml_type type_k : kv_types) {
            for (const ggml_type type_v : kv_types) {
                if (ggml_backend_cuda_kv_stream_get_attention_mode(type_k, type_v) !=
                        GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16) {
                    continue;
                }

                const attention_inputs inputs =
                    make_inputs(n_kv, n_batch, n_kv - n_batch, type_k, type_v);
                const attention_inputs reference = make_f16_reference(inputs, n_kv);
                const std::vector<float> expected = run_attention(
                    backend.get(), reference, ggml_backend_get_default_buffer_type(backend.get()),
                    n_kv, n_batch);

                const size_t k_page_bytes =
                    ggml_row_size(type_k, HEAD_DIM)*N_KV_HEAD*256;
                const size_t v_page_bytes =
                    ggml_row_size(type_v, HEAD_DIM)*N_KV_HEAD*256;
                const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
                const size_t f16_k_page_bytes =
                    ggml_row_size(GGML_TYPE_F16, HEAD_DIM)*N_KV_HEAD*256;
                const size_t f16_v_page_bytes =
                    ggml_row_size(GGML_TYPE_F16, HEAD_DIM)*N_KV_HEAD*256;
                const size_t conversion_bytes =
                    align_up(f16_k_page_bytes, 128) + f16_v_page_bytes;

                ggml_backend_cuda_kv_stream_params params{};
                params.device           = 0;
                params.stage_bytes      = page_bytes;
                params.stage_slots      = 1;
                params.conversion_bytes = conversion_bytes;
                auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
                if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
                    return;
                }

                const std::vector<float> actual = run_attention(
                    backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
                    n_kv, n_batch);
                const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
                ggml_backend_cuda_kv_stream_runtime_free(runtime);

                if (!t.assert_equal(expected.size(), actual.size())) {
                    return;
                }
                float max_abs = 0.0f;
                for (size_t i = 0; i < expected.size(); ++i) {
                    max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
                }
                if (!std::isfinite(max_abs) || max_abs > 2e-3f ||
                        stats.asynchronous_page_uploads == 0) {
                    std::fprintf(stderr,
                        "fallback pair K=%s V=%s max_abs=%g async_uploads=%llu\n",
                        ggml_type_name(type_k), ggml_type_name(type_v), max_abs,
                        (unsigned long long) stats.asynchronous_page_uploads);
                }
                t.assert_true("fallback pair executes streamed attention",
                    stats.asynchronous_page_uploads > 0);
                t.assert_true("fallback pair remains numerically equivalent",
                    std::isfinite(max_abs) && max_abs <= 2e-3f);
            }
        }
    });

    t.test("wide generic fallback remains equivalent across query workspace tiles", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        constexpr int64_t n_batch = 513;
        constexpr ggml_type type_k = GGML_TYPE_IQ4_NL;
        constexpr ggml_type type_v = GGML_TYPE_IQ4_NL;

        t.assert_equal(
            GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16,
            ggml_backend_cuda_kv_stream_get_attention_mode(type_k, type_v));
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs =
            make_inputs(n_kv, n_batch, n_kv - n_batch, type_k, type_v);
        const attention_inputs reference = make_f16_reference(inputs, n_kv);
        const std::vector<float> expected = run_attention(
            backend.get(), reference, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch);

        const size_t k_page_bytes = ggml_row_size(type_k, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(type_v, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        const size_t f16_page_bytes =
            ggml_row_size(GGML_TYPE_F16, HEAD_DIM)*N_KV_HEAD*256;
        const size_t conversion_bytes = align_up(f16_page_bytes, 128) + f16_page_bytes;

        ggml_backend_cuda_kv_stream_params params{};
        params.device           = 0;
        params.stage_bytes      = page_bytes;
        params.stage_slots      = 2;
        params.conversion_bytes = conversion_bytes;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        float max_abs = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
        }
        std::fprintf(stderr,
            "wide-fallback n_batch=%lld max_abs=%g async_uploads=%llu\n",
            (long long) n_batch, max_abs,
            (unsigned long long) stats.asynchronous_page_uploads);
        t.assert_true("wide fallback executes streamed attention",
            stats.asynchronous_page_uploads > 0);
        t.assert_equal(uint64_t(8), stats.host_to_device_copy_commands);
        t.assert_true("wide fallback remains numerically equivalent",
            std::isfinite(max_abs) && max_abs <= 2e-3f);
    });

    t.test("server-shaped causal prefill pipelines four Q8/Q4 blocks through two slots", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        constexpr int64_t n_batch = 83;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        // Exercise the final causal block. With query_start == 0, every block
        // after the first is masked and corrupted streamed pages are invisible.
        const attention_inputs inputs = make_inputs(n_kv, n_batch, n_kv - 256);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch, 2, 256, true);

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        std::vector<uint8_t> cleared(params.stage_bytes, 0);
        GGML_ASSERT(ggml_backend_cuda_kv_stream_stage_upload(
            runtime, 0, 0, cleared.data(), cleared.size()));

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch, 2, 256, true);


        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        t.assert_true("multi-token streamed spans use MMA partial attention",
            stats.mma_prefill_attention_spans > 0);
        t.assert_equal(uint64_t(6), stats.asynchronous_page_uploads);
        t.assert_equal(uint64_t(16), stats.host_to_device_copy_commands);
        t.assert_equal(uint64_t(6), stats.compute_stream_waits);
        t.assert_equal(uint64_t(4), stats.stage_slot_reuses);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);
        t.assert_equal(uint64_t(4), stats.streamed_attention_spans);
        t.assert_equal(uint64_t(6), stats.streamed_pages_attended);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }

        float max_abs = 0.0f;
        float max_rel = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
            max_rel = std::max(max_rel, std::abs(expected[i] - actual[i])/(std::abs(expected[i]) + 1e-6f));
        }
        std::fprintf(stderr, "streamed attention max_abs=%g max_rel=%g\n", max_abs, max_rel);
        t.assert_true("outputs remain finite", std::isfinite(max_abs) && std::isfinite(max_rel));
        t.assert_true("streamed output is numerically equivalent", max_abs <= 3e-4f);
    });

    t.test("one-page causal prefill stays bit-identical to ordinary CUDA attention", [](testing & t) {
        constexpr int64_t n_kv = 256;
        constexpr int64_t n_batch = 5;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device      = 0;
        params.stage_bytes = page_bytes;
        params.stage_slots = 1;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(0), stats.staged_set_rows);
        t.assert_equal(expected.size(), actual.size());
        t.assert_true("one-page outputs are bit-identical", expected == actual);
    });

    t.test("fully resident multi-page prefill stays bit-identical to ordinary CUDA attention", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 4;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch, 340);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 1;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("resident runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(0), stats.streamed_pages);
        t.assert_equal(expected.size(), actual.size());
        t.assert_true("fully resident outputs are bit-identical", expected == actual);
    });

    t.test("four-query page-boundary prefill remains finite and equivalent", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 4;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch, 340);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("resident runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        bool all_finite = true;
        float max_abs = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            all_finite = all_finite && std::isfinite(actual[i]);
            max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
        }
        std::fprintf(stderr, "four-query page-boundary max_abs=%g\n", max_abs);
        t.assert_true("page-boundary output remains finite", all_finite);
        t.assert_true("page-boundary output remains equivalent", max_abs <= 3e-4f);
    });

    t.test("wide causal prefills remain equivalent across the 256-query boundary", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        const int64_t query_counts[] = { 257, 512, 513, 1024 };
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;

        for (const int64_t n_batch : query_counts) {
            const attention_inputs inputs = make_inputs(n_kv, n_batch, n_kv - n_batch);
            const std::vector<float> expected = run_attention(
                backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
                n_kv, n_batch, 1, n_batch);

            ggml_backend_cuda_kv_stream_params params{};
            params.device               = 0;
            params.stage_bytes          = page_bytes;
            params.stage_slots          = 2;
            params.pool_bytes           = 3*page_bytes;
            params.resident_layer_count = 1;
            params.page_tokens          = 256;
            auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
            if (!t.assert_true("stream runtime initializes", runtime != nullptr)) {
                return;
            }

            const std::vector<float> actual = run_attention(
                backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
                n_kv, n_batch, 1, n_batch, false, GGML_TYPE_I64, false, runtime);
            const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
            ggml_backend_cuda_kv_stream_runtime_free(runtime);

            if (!t.assert_equal(expected.size(), actual.size())) {
                return;
            }
            bool all_finite = true;
            float max_abs = 0.0f;
            for (size_t i = 0; i < expected.size(); ++i) {
                all_finite = all_finite && std::isfinite(actual[i]);
                max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
            }
            std::fprintf(stderr,
                "wide-query n_batch=%lld max_abs=%g streamed_pages=%llu\n",
                (long long) n_batch, max_abs,
                (unsigned long long) stats.streamed_pages);
            t.assert_true("wide-query output remains finite", all_finite);
            t.assert_true("wide-query output remains equivalent", max_abs <= 3e-4f);
            t.assert_true("wide-query test exercises streamed pages", stats.streamed_pages > 0);
        }
    });

    t.test("resident pages survive between evaluations while the tail is refreshed", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;

        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 4;
        params.pool_bytes           = 6*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("resident runtime initializes", runtime != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(512, 256);
        (void) run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            512, 256, 2, 256, false, GGML_TYPE_I64, false, runtime);

        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        t.assert_equal(uint64_t(2), stats.resident_misses);
        t.assert_equal(uint64_t(2), stats.resident_hits);
        t.assert_equal(uint64_t(0), stats.streamed_pages);
        t.assert_equal(uint64_t(2*page_bytes), stats.host_to_device_bytes);
        t.assert_equal(uint64_t(2), stats.resident_attention_spans);
        t.assert_equal(uint64_t(4), stats.resident_pages_attended);
        t.assert_equal(uint64_t(4), stats.staged_set_rows);
        t.assert_equal(uint64_t(2*page_bytes), stats.staged_set_rows_bytes);

        t.assert_true("one resident page is demoted into the ring",
            ggml_backend_cuda_kv_stream_repartition(runtime, 5));
        t.assert_equal(uint32_t(1),
            ggml_backend_cuda_kv_stream_resident_pages_per_layer(runtime));
        (void) run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), 512, 256);

        const auto repartitioned = ggml_backend_cuda_kv_stream_get_stats(runtime);
        t.assert_equal(uint64_t(1), repartitioned.resident_misses);
        t.assert_equal(uint64_t(1), repartitioned.streamed_pages);
        t.assert_equal(uint64_t(1), repartitioned.asynchronous_page_uploads);
        t.assert_equal(uint64_t(1), repartitioned.resident_attention_spans);
        t.assert_equal(uint64_t(1), repartitioned.resident_pages_attended);

        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    t.test("one-row decode refreshes only the changed resident K and V rows", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 2, 1, true, GGML_TYPE_I32, true);

        const size_t k_token_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD;
        const size_t v_token_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD;
        const size_t k_page_bytes = k_token_bytes*256;
        const size_t v_page_bytes = v_token_bytes*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 1;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("resident runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 2, 1, true, GGML_TYPE_I32, true, runtime);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(2), stats.resident_misses);
        t.assert_equal(uint64_t(2), stats.resident_hits);
        t.assert_equal(
            uint64_t(2*page_bytes + k_token_bytes + v_token_bytes),
            stats.host_to_device_bytes);
        t.assert_equal(expected.size(), actual.size());
        t.assert_true("one-row resident refresh remains bit-identical", expected == actual);
    });

    t.test("resident SET_ROWS mirrors changing decode slots without page refreshes", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 5, 1, true, GGML_TYPE_I64, false, nullptr, true);

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 1;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("resident runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 5, 1, true, GGML_TYPE_I64, false, runtime, true);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);

        using feedback_fn_t = bool (*)(
            void *, uint64_t *, uint64_t *, double *, uint32_t *,
            uint32_t *, uint32_t *, uint32_t *);
        ggml_backend_dev_t device = ggml_backend_get_device(backend.get());
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        auto feedback_fn = reinterpret_cast<feedback_fn_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_stream_feedback"));
        uint64_t deadline_samples = 0;
        uint64_t deadline_misses = 0;
        double copy_busy_ratio = -1.0;
        uint32_t peak_occupancy = 0;
        uint32_t ring_slots = 0;
        uint32_t resident_pages = 0;
        uint32_t controlled_pages = 0;
        t.assert_true("feedback remains readable after resident graph replay",
            feedback_fn != nullptr && feedback_fn(
                runtime, &deadline_samples, &deadline_misses, &copy_busy_ratio,
                &peak_occupancy, &ring_slots, &resident_pages, &controlled_pages));
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(2*page_bytes), stats.host_to_device_bytes);
        t.assert_equal(expected.size(), actual.size());
        t.assert_true("mirrored changing slots remain bit-identical", expected == actual);
    });

    t.test("resident staged writes support every exposed KV-cache format", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 1;
        const ggml_type kv_types[] = {
            GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
            GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
        };
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        for (const ggml_type type : kv_types) {
            const attention_inputs inputs =
                make_inputs(n_kv, n_batch, n_kv - 1, type, type);
            const size_t typed_page_bytes =
                ggml_row_size(type, HEAD_DIM)*N_KV_HEAD*256;
            const size_t page_bytes = align_up(typed_page_bytes, 128) + typed_page_bytes;
            const bool fallback =
                ggml_backend_cuda_kv_stream_get_attention_mode(type, type) ==
                GGML_BACKEND_CUDA_KV_STREAM_ATTENTION_F16;
            const size_t f16_page_bytes =
                ggml_row_size(GGML_TYPE_F16, HEAD_DIM)*N_KV_HEAD*256;
            const size_t conversion_bytes = fallback ?
                align_up(f16_page_bytes, 128) + f16_page_bytes : 0;

            ggml_backend_cuda_kv_stream_params baseline_params{};
            baseline_params.device           = 0;
            baseline_params.stage_bytes      = page_bytes;
            baseline_params.stage_slots      = 1;
            baseline_params.conversion_bytes = conversion_bytes;
            auto baseline_runtime = ggml_backend_cuda_kv_stream_runtime_new(baseline_params);
            if (!t.assert_true("baseline runtime initializes", baseline_runtime != nullptr)) {
                return;
            }
            const std::vector<float> expected = run_attention(
                backend.get(), inputs,
                ggml_backend_cuda_kv_stream_buffer_type(baseline_runtime),
                n_kv, n_batch, 5, 3, true, GGML_TYPE_I64, false, nullptr, true);
            ggml_backend_cuda_kv_stream_runtime_free(baseline_runtime);

            ggml_backend_cuda_kv_stream_params resident_params{};
            resident_params.device               = 0;
            resident_params.stage_bytes          = page_bytes;
            resident_params.stage_slots          = 1;
            resident_params.pool_bytes           = 3*page_bytes + conversion_bytes;
            resident_params.conversion_bytes     = conversion_bytes;
            resident_params.resident_layer_count = 1;
            resident_params.page_tokens          = 256;
            auto resident_runtime = ggml_backend_cuda_kv_stream_runtime_new(resident_params);
            if (!t.assert_true("resident runtime initializes", resident_runtime != nullptr)) {
                return;
            }
            const std::vector<float> actual = run_attention(
                backend.get(), inputs,
                ggml_backend_cuda_kv_stream_buffer_type(resident_runtime),
                n_kv, n_batch, 5, 3, true, GGML_TYPE_I64, false,
                resident_runtime, true);
            const auto stats = ggml_backend_cuda_kv_stream_get_stats(resident_runtime);
            ggml_backend_cuda_kv_stream_runtime_free(resident_runtime);

            if (!t.assert_equal(expected.size(), actual.size())) {
                return;
            }
            float max_abs = 0.0f;
            for (size_t i = 0; i < expected.size(); ++i) {
                max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
            }
            if (!t.assert_true(ggml_type_name(type),
                    std::isfinite(max_abs) && max_abs <= 5e-4f)) {
                std::fprintf(stderr, "resident staged write type=%s max_abs=%g\n",
                    ggml_type_name(type), max_abs);
                return;
            }
            t.assert_equal(uint64_t(2*page_bytes), stats.host_to_device_bytes);
            t.assert_equal(uint64_t(10), stats.staged_set_rows);
            t.assert_equal(
                uint64_t(10*3*ggml_row_size(type, HEAD_DIM*N_KV_HEAD)),
                stats.staged_set_rows_bytes);
        }
    });

    t.test("resident graph reloads after authoritative cache replacement", [](testing & t) {
        constexpr int64_t n_kv = 512;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 5, 1, true, GGML_TYPE_I64, false, nullptr, true, true, 1);

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 1;
        params.pool_bytes           = 3*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("resident runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 5, 1, true, GGML_TYPE_I64, false, runtime, true, true, 1);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(2), stats.resident_misses);
        t.assert_equal(expected.size(), actual.size());
        t.assert_true("reloaded resident output is bit-identical", expected == actual);
    });

    t.test("decode batches contiguous streamed pages without changing logits", [](testing & t) {
        constexpr int64_t n_kv = 1536;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 2);

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 4;
        params.pool_bytes           = 5*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("batched runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 2);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(10), stats.asynchronous_page_uploads);
        t.assert_equal(uint64_t(12), stats.host_to_device_copy_commands);
        t.assert_equal(uint64_t(4), stats.compute_stream_waits);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        float max_abs = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
        }
        t.assert_true("batched streamed logits remain equivalent", max_abs <= 3e-4f);
    });

    t.test("decode pipelines bounded attention chunks and samples every immutable copy batch", [](testing & t) {
        constexpr int64_t n_kv = 41*256;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const attention_inputs inputs = make_inputs(n_kv, n_batch);
        const std::vector<float> expected = run_attention(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch);

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 40;
        params.pool_bytes           = 41*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = 256;
        params.decode_span_pages    = 32;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("chunk-pipelined runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(40), stats.streamed_pages);
        t.assert_equal(uint64_t(2), stats.streamed_attention_spans);
        // The 39 immutable pages form one full 32-page transfer and one
        // seven-page transfer. The mutable tail is deliberately excluded from
        // adaptive prefetch feedback because its producer runs in this graph.
        t.assert_equal(uint64_t(2), stats.deadline_samples);
        t.assert_true("chunk deadline misses cannot exceed samples",
            stats.deadline_misses <= stats.deadline_samples);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        float max_abs = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
        }
        t.assert_true("chunk-pipelined logits remain equivalent", max_abs <= 3e-4f);
    });

    t.test("sixteen attention layers share one resident/ring pool during causal prefill", [](testing & t) {
        constexpr int64_t n_kv = 1024;
        constexpr int64_t n_batch = 83;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        std::vector<attention_inputs> inputs(16, make_inputs(n_kv, n_batch, n_kv - 256));
        for (size_t layer = 1; layer < inputs.size(); ++layer) {
            for (size_t i = 0; i < inputs[layer].q.size(); ++i) {
                inputs[layer].q[i] += 0.025f*layer*std::sin(float(i)*0.015625f);
            }
        }
        const std::vector<float> expected = run_attention_layers(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()),
            n_kv, n_batch, 1, 2, GGML_TYPE_I64);

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 8;
        params.pool_bytes           = 24*page_bytes;
        params.resident_layer_count = 16;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 1, 2, GGML_TYPE_I64, runtime);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        t.assert_equal(uint64_t(32), stats.staged_set_rows);
        t.assert_equal(
            uint64_t(16*(2*ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM*N_KV_HEAD) +
                         2*ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM*N_KV_HEAD))),
            stats.staged_set_rows_bytes);
        t.assert_equal(uint64_t(48), stats.asynchronous_page_uploads);
        t.assert_equal(uint64_t(48), stats.compute_stream_waits);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(expected.size(), actual.size());
        float max_abs = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
        }
        t.assert_true("sixteen-layer prefill remains equivalent", max_abs <= 3e-4f);
    });

    t.test("one shared ring prefetches across attention layers", [](testing & t) {
        constexpr int64_t n_kv = 768;
        // Cross-layer prefetch is a decode optimization. A one-token batch
        // also guarantees that only the mutable tail page is changed by the
        // SET_ROWS producers in this graph.
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        std::vector<attention_inputs> inputs{
            make_inputs(n_kv, n_batch),
            make_inputs(n_kv, n_batch),
            make_inputs(n_kv, n_batch),
        };
        for (size_t layer = 1; layer < inputs.size(); ++layer) {
            for (size_t i = 0; i < inputs[layer].q.size(); ++i) {
                inputs[layer].q[i] += 0.025f*layer*std::sin(float(i)*0.015625f);
            }
        }
        const std::vector<float> expected = run_attention_layers(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 5*page_bytes;
        params.resident_layer_count = 3;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }

        const std::vector<float> actual = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime), n_kv, n_batch, 2);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        t.assert_equal(uint64_t(12), stats.asynchronous_page_uploads);
        t.assert_equal(uint64_t(12), stats.compute_stream_waits);
        t.assert_equal(uint64_t(4), stats.cross_layer_prefetches);
        t.assert_equal(uint64_t(6), stats.deadline_samples);
        t.assert_true("deadline misses cannot exceed samples",
            stats.deadline_misses <= stats.deadline_samples);
        t.assert_equal(uint32_t(2), stats.ring_peak_occupancy);
        t.assert_equal(uint64_t(6), stats.streamed_attention_spans);
        t.assert_equal(uint64_t(12), stats.streamed_pages_attended);

        using feedback_fn_t = bool (*)(
            void *, uint64_t *, uint64_t *, double *, uint32_t *,
            uint32_t *, uint32_t *, uint32_t *);
        ggml_backend_dev_t device = ggml_backend_get_device(backend.get());
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        auto feedback_fn = reinterpret_cast<feedback_fn_t>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_kv_stream_feedback"));
        if (!t.assert_true("dynamic feedback API is exported", feedback_fn != nullptr)) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }
        uint64_t deadline_samples = 0;
        uint64_t deadline_misses = 0;
        double copy_busy_ratio = -1.0;
        uint32_t peak_occupancy = 0;
        uint32_t ring_slots = 0;
        uint32_t resident_pages = 0;
        uint32_t controlled_pages = 0;
        t.assert_true("dynamic feedback is readable", feedback_fn(
            runtime, &deadline_samples, &deadline_misses, &copy_busy_ratio,
            &peak_occupancy, &ring_slots, &resident_pages, &controlled_pages));
        t.assert_equal(uint64_t(6), deadline_samples);
        t.assert_true("copy busy ratio is normalized",
            copy_busy_ratio >= 0.0 && copy_busy_ratio <= 1.0);
        t.assert_equal(uint32_t(2), peak_occupancy);
        t.assert_equal(uint32_t(2), ring_slots);
        t.assert_equal(uint32_t(1), resident_pages);
        t.assert_equal(uint32_t(5), controlled_pages);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        float max_abs = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
        }
        std::fprintf(stderr, "multi-layer streamed attention max_abs=%g\n", max_abs);
        t.assert_true("multi-layer streamed logits remain equivalent", max_abs <= 3e-4f);
    });

    t.test("decode layout permits multiple streaming waves through a small ring", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 1;
        params.pool_bytes           = 5*page_bytes;
        params.resident_layer_count = 4;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("small-ring runtime initializes", runtime != nullptr)) {
            return;
        }

        // Each of four layers has one resident and two streamed pages. The
        // one-slot ring services each layer in two waves, so requiring one
        // split layer per ring-sized portion would incorrectly demand eight
        // split layers from a four-layer model.
        t.assert_true("small ring selects a valid decode layout",
            ggml_backend_cuda_kv_stream_set_decode_layout(runtime, 3));
        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    t.test("decode layout and ring boundary publish as one reconfiguration", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 8;
        params.pool_bytes           = 12*page_bytes;
        params.resident_layer_count = 4;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }

        t.assert_true("combined reconfiguration succeeds",
            ggml_backend_cuda_kv_stream_reconfigure(
                runtime, /* active pages per layer = */ 3, /* ring slots = */ 4));
        t.assert_equal(uint32_t(4),
            ggml_backend_cuda_kv_stream_stage_slots(runtime));
        t.assert_equal(uint32_t(2),
            ggml_backend_cuda_kv_stream_resident_pages_per_layer(runtime));

        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });


    t.test("decode layout spreads an oversized streamed deficit across enough layers", [](testing & t) {
        constexpr int64_t n_kv = 768;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        std::vector<attention_inputs> inputs{
            make_inputs(n_kv, n_batch, n_kv - 1),
            make_inputs(n_kv, n_batch, n_kv - 1),
            make_inputs(n_kv, n_batch, n_kv - 1),
            make_inputs(n_kv, n_batch, n_kv - 1),
        };
        const std::vector<float> expected = run_attention_layers(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 8;
        params.pool_bytes           = 12*page_bytes;
        params.resident_layer_count = 4;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }

        // Four layers with three active pages and one uniformly resident page
        // have an eight-page streamed deficit. An eight-slot ring can hold the
        // entire deficit, but one layer contains only three pages, so the layout
        // must distribute the deficit over at least ceil(8/3) layers.
        if (!t.assert_true("decode layout respects per-layer active-page capacity",
                ggml_backend_cuda_kv_stream_set_decode_layout(runtime, 3))) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }

        const std::vector<float> actual = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 1, 1, GGML_TYPE_I32, runtime);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(8), stats.streamed_pages);
        t.assert_equal(uint64_t(4), stats.resident_pages_attended);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        float max_abs = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(expected[i] - actual[i]));
        }
        std::fprintf(stderr, "oversized-ring decode max_abs=%g\n", max_abs);
        t.assert_true("oversized-ring decode remains equivalent", max_abs <= 3e-4f);
    });

    t.test("decode layout bounds layer concentration by the transfer ring", [](testing & t) {
        constexpr int64_t n_kv = 768;
        constexpr int64_t n_batch = 1;
        ggml_backend_ptr backend(ggml_backend_cuda_init(0));
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }

        std::vector<attention_inputs> inputs{
            make_inputs(n_kv, n_batch, n_kv - 1),
            make_inputs(n_kv, n_batch, n_kv - 1),
            make_inputs(n_kv, n_batch, n_kv - 1),
        };
        const std::vector<float> expected = run_attention_layers(
            backend.get(), inputs, ggml_backend_get_default_buffer_type(backend.get()), n_kv, n_batch);

        const size_t k_page_bytes = ggml_row_size(GGML_TYPE_Q8_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t v_page_bytes = ggml_row_size(GGML_TYPE_Q4_0, HEAD_DIM)*N_KV_HEAD*256;
        const size_t page_bytes = align_up(k_page_bytes, 128) + v_page_bytes;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 3;
        params.pool_bytes           = 6*page_bytes;
        params.resident_layer_count = 3;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("shared runtime initializes", runtime != nullptr)) {
            return;
        }
        const std::vector<float> uniform = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 1, 1, GGML_TYPE_I32, runtime);
        float uniform_max_abs = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            uniform_max_abs =
                std::max(uniform_max_abs, std::abs(expected[i] - uniform[i]));
        }
        std::fprintf(stderr, "uniform decode max_abs=%g\n", uniform_max_abs);
        t.assert_true("uniform decode control remains equivalent", uniform_max_abs <= 3e-4f);
        if (!t.assert_true("decode residency becomes ring-bounded",
                ggml_backend_cuda_kv_stream_set_decode_layout(runtime, 3))) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }

        const std::vector<float> actual = run_attention_layers(
            backend.get(), inputs, ggml_backend_cuda_kv_stream_buffer_type(runtime),
            n_kv, n_batch, 2, 1, GGML_TYPE_I32, runtime, 4);
        const auto stats = ggml_backend_cuda_kv_stream_get_stats(runtime);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);

        t.assert_equal(uint64_t(6), stats.resident_misses);
        t.assert_equal(uint64_t(0), stats.resident_hits);
        t.assert_equal(uint64_t(12), stats.streamed_pages);
        t.assert_equal(uint64_t(4), stats.resident_attention_spans);
        t.assert_equal(uint64_t(6), stats.resident_pages_attended);
        if (!t.assert_equal(expected.size(), actual.size())) {
            return;
        }
        float max_abs = 0.0f;
        std::vector<float> layer_max_abs(inputs.size(), 0.0f);
        const size_t layer_elements = expected.size()/inputs.size();
        for (size_t i = 0; i < expected.size(); ++i) {
            const float error = std::abs(expected[i] - actual[i]);
            max_abs = std::max(max_abs, error);
            layer_max_abs[i/layer_elements] = std::max(layer_max_abs[i/layer_elements], error);
        }
        std::fprintf(stderr, "ring-bounded decode max_abs=%g layers=%g,%g,%g\n",
            max_abs, layer_max_abs[0], layer_max_abs[1], layer_max_abs[2]);
        t.assert_true("ring-bounded decode remains equivalent", max_abs <= 3e-4f);
    });

    ggml_quantize_free();
    return t.summary();
}
