#include "ggml-cuda.h"
#include "testing.h"

#include <cstdint>
#include <vector>

int main() {
    testing t;

    t.test("runtime exposes exact staging geometry", [](testing & t) {
        ggml_backend_cuda_kv_stream_params params{};
        params.device      = 0;
        params.stage_bytes = 1024*1024;
        params.stage_slots = 2;

        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("runtime allocation succeeds", runtime != nullptr)) {
            return;
        }

        t.assert_equal(params.stage_bytes, ggml_backend_cuda_kv_stream_stage_bytes(runtime));
        t.assert_equal(params.stage_slots, ggml_backend_cuda_kv_stream_stage_slots(runtime));
        auto buft = ggml_backend_cuda_kv_stream_buffer_type(runtime);
        if (!t.assert_true("streamed buffer type exists", buft != nullptr)) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }
        t.assert_true(
            "owning CUDA device accepts streamed buffers",
            ggml_backend_dev_supports_buft(ggml_backend_buft_get_device(buft), buft));
        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    t.test("authoritative buffer is pinned-host accessible", [](testing & t) {
        ggml_backend_cuda_kv_stream_params params{};
        params.device      = 0;
        params.stage_bytes = 1024*1024;
        params.stage_slots = 1;

        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("runtime allocation succeeds", runtime != nullptr)) {
            return;
        }

        auto buft = ggml_backend_cuda_kv_stream_buffer_type(runtime);
        auto buffer = ggml_backend_buft_alloc_buffer(buft, 256*1024);
        if (!t.assert_true("host buffer allocation succeeds", buffer != nullptr)) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }

        t.assert_true("buffer reports host accessibility", ggml_backend_buffer_is_host(buffer));
        t.assert_equal(size_t(256*1024), ggml_backend_buffer_get_size(buffer));
        void * base = ggml_backend_buffer_get_base(buffer);
        t.assert_true("buffer base exists", base != nullptr);

        using query_fn_t = bool (*)(const void *, bool *);
        ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        auto query_fn = reinterpret_cast<query_fn_t>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_kv_stream_host_is_write_combined"));
        if (!t.assert_true("host allocation query is exported", query_fn != nullptr)) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            ggml_backend_buffer_free(buffer);
            return;
        }
        bool write_combined = false;
        t.assert_true("host allocation flags are readable",
            query_fn(base, &write_combined));
#if defined(_WIN32)
        t.assert_true("authoritative KV storage is not write-combined on Windows", !write_combined);
#else
        t.assert_true("authoritative KV storage is write-combined", write_combined);
#endif

        ggml_backend_cuda_kv_stream_runtime_free(runtime);
        ggml_backend_buffer_free(buffer);
    });

    t.test("invalid runtime geometry is rejected", [](testing & t) {
        ggml_backend_cuda_kv_stream_params params{};
        params.device      = -1;
        params.stage_bytes = 1024*1024;
        params.stage_slots = 1;
        t.assert_true("negative device is rejected", ggml_backend_cuda_kv_stream_runtime_new(params) == nullptr);

        params.device      = 0;
        params.stage_bytes = 0;
        t.assert_true("zero stage size is rejected", ggml_backend_cuda_kv_stream_runtime_new(params) == nullptr);

        params.stage_bytes = 1024*1024;
        params.stage_slots = 0;
        t.assert_true("zero stage slots are rejected", ggml_backend_cuda_kv_stream_runtime_new(params) == nullptr);
    });

    t.test("unimplemented operations cannot consume streamed storage", [](testing & t) {
        ggml_backend_cuda_kv_stream_params params{};
        params.device      = 0;
        params.stage_bytes = 1024*1024;
        params.stage_slots = 1;

        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("runtime allocation succeeds", runtime != nullptr)) {
            return;
        }

        auto buft = ggml_backend_cuda_kv_stream_buffer_type(runtime);
        auto buffer = ggml_backend_buft_alloc_buffer(buft, 4096);
        if (!t.assert_true("host buffer allocation succeeds", buffer != nullptr)) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            return;
        }

        ggml_tensor source{};
        source.type   = GGML_TYPE_F32;
        source.buffer = buffer;
        source.data   = ggml_backend_buffer_get_base(buffer);
        source.ne[0]  = 1;
        source.ne[1]  = 1;
        source.ne[2]  = 1;
        source.ne[3]  = 1;
        source.nb[0]  = sizeof(float);
        source.nb[1]  = sizeof(float);
        source.nb[2]  = sizeof(float);
        source.nb[3]  = sizeof(float);

        ggml_tensor fill{};
        fill.op   = GGML_OP_FILL;
        fill.type = GGML_TYPE_F32;
        fill.ne[0] = fill.ne[1] = fill.ne[2] = fill.ne[3] = 1;
        fill.nb[0] = fill.nb[1] = fill.nb[2] = fill.nb[3] = sizeof(float);

        auto device = ggml_backend_buft_get_device(buft);
        t.assert_true("ordinary fill is supported", ggml_backend_dev_supports_op(device, &fill));
        fill.src[0] = &source;
        t.assert_true("fill from streamed storage is rejected", !ggml_backend_dev_supports_op(device, &fill));

        ggml_backend_buffer_free(buffer);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    t.test("stage slots round-trip bounded byte ranges", [](testing & t) {
        ggml_backend_cuda_kv_stream_params params{};
        params.device      = 0;
        params.stage_bytes = 64*1024;
        params.stage_slots = 2;

        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("runtime allocation succeeds", runtime != nullptr)) {
            return;
        }

        std::vector<uint8_t> source(8192);
        for (size_t i = 0; i < source.size(); ++i) {
            source[i] = uint8_t((i*37 + 11) & 0xff);
        }
        std::vector<uint8_t> destination(source.size(), 0);

        t.assert_true("bounded upload succeeds", ggml_backend_cuda_kv_stream_stage_upload(
            runtime, 1, 123, source.data(), source.size()));
        t.assert_true("bounded download succeeds", ggml_backend_cuda_kv_stream_stage_download(
            runtime, 1, 123, destination.data(), destination.size()));
        t.assert_true("round-trip bytes are exact", source == destination);

        t.assert_true("invalid slot is rejected", !ggml_backend_cuda_kv_stream_stage_upload(
            runtime, 2, 0, source.data(), source.size()));
        t.assert_true("cross-slot range is rejected", !ggml_backend_cuda_kv_stream_stage_upload(
            runtime, 0, params.stage_bytes - 16, source.data(), source.size()));
        t.assert_true("null source is rejected", !ggml_backend_cuda_kv_stream_stage_upload(
            runtime, 0, 0, nullptr, source.size()));

        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    t.test("fixed pool moves pages from balanced residency into the active ring", [](testing & t) {
        constexpr size_t page_bytes = 64*1024;
        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 6*page_bytes;
        params.resident_layer_count = 1;
        params.page_tokens          = 256;

        auto runtime = ggml_backend_cuda_kv_stream_runtime_new(params);
        if (!t.assert_true("runtime allocation succeeds", runtime != nullptr)) {
            return;
        }

        t.assert_equal(uint32_t(2), ggml_backend_cuda_kv_stream_stage_slots(runtime));
        t.assert_equal(uint32_t(4), ggml_backend_cuda_kv_stream_resident_pages_per_layer(runtime));
        t.assert_true("ring grows inside the existing pool",
            ggml_backend_cuda_kv_stream_repartition(runtime, 3));
        t.assert_equal(uint32_t(3), ggml_backend_cuda_kv_stream_stage_slots(runtime));
        t.assert_equal(uint32_t(3), ggml_backend_cuda_kv_stream_resident_pages_per_layer(runtime));
        t.assert_true("ring cannot exceed the fixed pool",
            !ggml_backend_cuda_kv_stream_repartition(runtime, 7));

        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    t.test("device factory assigns unusable resident remainder pages to the ring", [](testing & t) {
        constexpr size_t page_bytes = 64*1024;
        constexpr size_t pool_pages = 160;
        constexpr uint32_t layers = 16;
        using new_fn_t = void * (*)(ggml_backend_dev_t, size_t, size_t, size_t, uint32_t);

        ggml_backend_t backend = ggml_backend_cuda_init(0);
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        auto new_fn = reinterpret_cast<new_fn_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device),
            "ggml_backend_cuda_kv_stream_runtime_new_for_device"));
        if (!t.assert_true("device factory is exported", new_fn != nullptr)) {
            ggml_backend_free(backend);
            return;
        }
        auto runtime = static_cast<ggml_backend_cuda_kv_stream_runtime_t>(
            new_fn(device, pool_pages*page_bytes, page_bytes, 0, layers));
        if (!t.assert_true("factory runtime initializes", runtime != nullptr)) {
            ggml_backend_free(backend);
            return;
        }

        t.assert_equal(uint32_t(16), ggml_backend_cuda_kv_stream_stage_slots(runtime));
        t.assert_equal(uint32_t(9),
            ggml_backend_cuda_kv_stream_resident_pages_per_layer(runtime));

        ggml_backend_cuda_kv_stream_runtime_free(runtime);
        ggml_backend_free(backend);
    });

    t.test("phase arena lends an exact CUDA compute slice", [](testing & t) {
        constexpr size_t MiB = 1024*1024;
        auto arena = ggml_backend_cuda_phase_arena_new(0, 4*MiB);
        if (!t.assert_true("arena allocation succeeds", arena != nullptr)) {
            return;
        }
        t.assert_equal(size_t(4*MiB), ggml_backend_cuda_phase_arena_size(arena));
        t.assert_true("compute slice is accepted",
            ggml_backend_cuda_phase_arena_set_compute(arena, MiB, 2*MiB));

        auto buft = ggml_backend_cuda_phase_arena_buffer_type(arena);
        if (!t.assert_true("borrowed CUDA buffer type exists", buft != nullptr)) {
            ggml_backend_cuda_phase_arena_free(arena);
            return;
        }
        t.assert_true("CUDA device accepts borrowed buffer type",
            ggml_backend_dev_supports_buft(ggml_backend_buft_get_device(buft), buft));
        t.assert_true("oversized lease is rejected",
            ggml_backend_buft_alloc_buffer(buft, 2*MiB + 1) == nullptr);

        auto buffer = ggml_backend_buft_alloc_buffer(buft, 2*MiB);
        if (!t.assert_true("exact compute lease succeeds", buffer != nullptr)) {
            ggml_backend_cuda_phase_arena_free(arena);
            return;
        }
        t.assert_equal(size_t(2*MiB), ggml_backend_buffer_get_size(buffer));
        t.assert_true("borrowed base exists", ggml_backend_buffer_get_base(buffer) != nullptr);
        t.assert_true("concurrent lease is rejected",
            ggml_backend_buft_alloc_buffer(buft, MiB) == nullptr);

        ggml_backend_buffer_free(buffer);
        ggml_backend_cuda_phase_arena_free(arena);
    });

    t.test("borrowed compute buffer keeps its arena alive", [](testing & t) {
        constexpr size_t MiB = 1024*1024;
        auto arena = ggml_backend_cuda_phase_arena_new(0, 2*MiB);
        if (!t.assert_true("arena allocation succeeds", arena != nullptr)) {
            return;
        }
        t.assert_true("compute slice is accepted",
            ggml_backend_cuda_phase_arena_set_compute(arena, MiB, MiB));
        auto buffer = ggml_backend_buft_alloc_buffer(
            ggml_backend_cuda_phase_arena_buffer_type(arena), MiB);
        if (!t.assert_true("compute lease succeeds", buffer != nullptr)) {
            ggml_backend_cuda_phase_arena_free(arena);
            return;
        }

        ggml_tensor tensor{};
        tensor.type   = GGML_TYPE_I8;
        tensor.buffer = buffer;
        tensor.data   = ggml_backend_buffer_get_base(buffer);
        tensor.ne[0]  = 4096;
        tensor.ne[1]  = tensor.ne[2] = tensor.ne[3] = 1;
        tensor.nb[0]  = 1;
        tensor.nb[1]  = 4096;
        tensor.nb[2]  = tensor.nb[3] = 4096;

        std::vector<uint8_t> source(4096);
        for (size_t i = 0; i < source.size(); ++i) {
            source[i] = uint8_t((i*29 + 7) & 0xff);
        }
        std::vector<uint8_t> destination(source.size(), 0);

        ggml_backend_cuda_phase_arena_free(arena);
        ggml_backend_tensor_set(&tensor, source.data(), 0, source.size());
        ggml_backend_tensor_get(&tensor, destination.data(), 0, destination.size());
        t.assert_true("borrowed storage remains valid", source == destination);
        ggml_backend_buffer_free(buffer);
    });

    t.test("CUDA backend async copies accept borrowed compute buffers", [](testing & t) {
        constexpr size_t MiB = 1024*1024;
        auto backend = ggml_backend_cuda_init(0);
        auto arena = ggml_backend_cuda_phase_arena_new(0, 2*MiB);
        if (!t.assert_true("CUDA backend initializes", backend != nullptr) ||
                !t.assert_true("arena allocation succeeds", arena != nullptr)) {
            ggml_backend_free(backend);
            ggml_backend_cuda_phase_arena_free(arena);
            return;
        }
        t.assert_true("compute slice is accepted",
            ggml_backend_cuda_phase_arena_set_compute(arena, MiB, MiB));
        auto buffer = ggml_backend_buft_alloc_buffer(
            ggml_backend_cuda_phase_arena_buffer_type(arena), MiB);
        if (!t.assert_true("compute lease succeeds", buffer != nullptr)) {
            ggml_backend_cuda_phase_arena_free(arena);
            ggml_backend_free(backend);
            return;
        }

        ggml_tensor tensor{};
        tensor.type   = GGML_TYPE_I8;
        tensor.buffer = buffer;
        tensor.data   = ggml_backend_buffer_get_base(buffer);
        tensor.ne[0]  = 4096;
        tensor.ne[1]  = tensor.ne[2] = tensor.ne[3] = 1;
        tensor.nb[0]  = 1;
        tensor.nb[1]  = tensor.nb[2] = tensor.nb[3] = 4096;

        std::vector<uint8_t> source(4096, 0x5a);
        std::vector<uint8_t> destination(source.size(), 0);
        ggml_backend_tensor_set_async(
            backend, &tensor, source.data(), 0, source.size());
        ggml_backend_tensor_get_async(
            backend, &tensor, destination.data(), 0, destination.size());
        ggml_backend_synchronize(backend);
        t.assert_true("async round trip succeeds", source == destination);

        ggml_backend_buffer_free(buffer);
        ggml_backend_cuda_phase_arena_free(arena);
        ggml_backend_free(backend);
    });

    t.test("CUDA graph state can be invalidated before arena reuse", [](testing & t) {
        auto backend = ggml_backend_cuda_init(0);
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }
        t.assert_true("empty graph state resets",
            ggml_backend_cuda_graph_reset(backend));
        t.assert_true("null backend is rejected",
            !ggml_backend_cuda_graph_reset(nullptr));
        ggml_backend_free(backend);
    });

    t.test("phase arena can move its compute lease after release", [](testing & t) {
        constexpr size_t MiB = 1024*1024;
        auto arena = ggml_backend_cuda_phase_arena_new(0, 4*MiB);
        if (!t.assert_true("arena allocation succeeds", arena != nullptr)) {
            return;
        }
        auto buft = ggml_backend_cuda_phase_arena_buffer_type(arena);
        t.assert_true("first range is accepted",
            ggml_backend_cuda_phase_arena_set_compute(arena, MiB, MiB));
        auto first = ggml_backend_buft_alloc_buffer(buft, MiB);
        if (!t.assert_true("first lease succeeds", first != nullptr)) {
            ggml_backend_cuda_phase_arena_free(arena);
            return;
        }
        void * first_base = ggml_backend_buffer_get_base(first);
        t.assert_true("active lease cannot move",
            !ggml_backend_cuda_phase_arena_set_compute(arena, 2*MiB, MiB));
        ggml_backend_buffer_free(first);

        t.assert_true("released lease can move",
            ggml_backend_cuda_phase_arena_set_compute(arena, 2*MiB, MiB));
        auto second = ggml_backend_buft_alloc_buffer(buft, MiB);
        if (t.assert_true("second lease succeeds", second != nullptr)) {
            t.assert_true("lease base moved", ggml_backend_buffer_get_base(second) != first_base);
            ggml_backend_buffer_free(second);
        }
        ggml_backend_cuda_phase_arena_free(arena);
    });

    t.test("phase arena rejects invalid ranges", [](testing & t) {
        constexpr size_t MiB = 1024*1024;
        t.assert_true("zero arena is rejected",
            ggml_backend_cuda_phase_arena_new(0, 0) == nullptr);

        auto arena = ggml_backend_cuda_phase_arena_new(0, 2*MiB);
        if (!t.assert_true("arena allocation succeeds", arena != nullptr)) {
            return;
        }
        t.assert_true("zero compute size is rejected",
            !ggml_backend_cuda_phase_arena_set_compute(arena, 0, 0));
        t.assert_true("out of range offset is rejected",
            !ggml_backend_cuda_phase_arena_set_compute(arena, 3*MiB, 1));
        t.assert_true("overflowing range is rejected",
            !ggml_backend_cuda_phase_arena_set_compute(arena, 2*MiB - 1, 2));
        ggml_backend_cuda_phase_arena_free(arena);
    });

    t.test("phase arena operations are exported through the CUDA registry", [](testing & t) {
        ggml_backend_t backend = ggml_backend_cuda_init(0);
        if (!t.assert_true("CUDA backend initializes", backend != nullptr)) {
            return;
        }
        auto device = ggml_backend_get_device(backend);
        auto registry = ggml_backend_dev_backend_reg(device);
        const char * names[] = {
            "ggml_backend_cuda_phase_arena_new_for_device",
            "ggml_backend_cuda_phase_arena_free",
            "ggml_backend_cuda_phase_arena_size",
            "ggml_backend_cuda_phase_arena_set_compute",
            "ggml_backend_cuda_phase_arena_buffer_type",
            "ggml_backend_cuda_graph_reset",
            "ggml_backend_cuda_kv_stream_runtime_new_for_device_in_phase_arena",
            "ggml_backend_cuda_kv_stream_pool_bytes",
            "ggml_backend_cuda_kv_stream_resize_pool",
        };
        for (const char * name : names) {
            t.assert_true(name, ggml_backend_reg_get_proc_address(registry, name) != nullptr);
        }
        ggml_backend_free(backend);
    });

    t.test("KV runtime expands inside the shared arena after compute releases", [](testing & t) {
        constexpr size_t page_bytes = 64*1024;
        constexpr size_t arena_pages = 20;
        auto arena = ggml_backend_cuda_phase_arena_new(0, arena_pages*page_bytes);
        if (!t.assert_true("arena allocation succeeds", arena != nullptr)) {
            return;
        }
        t.assert_true("prefill compute range is accepted",
            ggml_backend_cuda_phase_arena_set_compute(
                arena, 12*page_bytes, 8*page_bytes));

        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 12*page_bytes;
        params.resident_layer_count = 2;
        params.page_tokens          = 256;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new_in_phase_arena(
            arena, 18*page_bytes, params);
        if (!t.assert_true("arena-backed KV runtime initializes", runtime != nullptr)) {
            ggml_backend_cuda_phase_arena_free(arena);
            return;
        }
        t.assert_equal(size_t(12*page_bytes),
            ggml_backend_cuda_kv_stream_pool_bytes(runtime));
        t.assert_equal(uint32_t(2), ggml_backend_cuda_kv_stream_stage_slots(runtime));
        t.assert_equal(uint32_t(5),
            ggml_backend_cuda_kv_stream_resident_pages_per_layer(runtime));

        auto compute = ggml_backend_buft_alloc_buffer(
            ggml_backend_cuda_phase_arena_buffer_type(arena), 8*page_bytes);
        if (!t.assert_true("prefill compute lease succeeds", compute != nullptr)) {
            ggml_backend_cuda_kv_stream_runtime_free(runtime);
            ggml_backend_cuda_phase_arena_free(arena);
            return;
        }
        std::vector<uint8_t> source(4096, 0x5a);
        std::vector<uint8_t> destination(source.size(), 0);
        t.assert_true("prefill ring remains usable",
            ggml_backend_cuda_kv_stream_stage_upload(
                runtime, 1, 0, source.data(), source.size()));
        t.assert_true("prefill ring round-trip succeeds",
            ggml_backend_cuda_kv_stream_stage_download(
                runtime, 1, 0, destination.data(), destination.size()));
        t.assert_true("prefill ring bytes are exact", source == destination);

        ggml_backend_buffer_free(compute);
        t.assert_true("decode compute range is accepted",
            ggml_backend_cuda_phase_arena_set_compute(
                arena, 18*page_bytes, 2*page_bytes));
        t.assert_true("KV pool expands and retains a nonzero ring",
            ggml_backend_cuda_kv_stream_resize_pool(
                runtime, 18*page_bytes, 0, 3));
        t.assert_equal(size_t(18*page_bytes),
            ggml_backend_cuda_kv_stream_pool_bytes(runtime));
        t.assert_equal(uint32_t(3), ggml_backend_cuda_kv_stream_stage_slots(runtime));
        t.assert_equal(uint32_t(7),
            ggml_backend_cuda_kv_stream_resident_pages_per_layer(runtime));

        auto decode_compute = ggml_backend_buft_alloc_buffer(
            ggml_backend_cuda_phase_arena_buffer_type(arena), 2*page_bytes);
        if (t.assert_true("decode compute lease succeeds", decode_compute != nullptr)) {
            ggml_backend_buffer_free(decode_compute);
        }
        t.assert_true("KV pool shrinks before returning to prefill",
            ggml_backend_cuda_kv_stream_resize_pool(
                runtime, 12*page_bytes, 0, 2));
        t.assert_true("prefill compute can move down after KV shrinks",
            ggml_backend_cuda_phase_arena_set_compute(
                arena, 12*page_bytes, 8*page_bytes));
        t.assert_equal(uint32_t(2), ggml_backend_cuda_kv_stream_stage_slots(runtime));
        t.assert_equal(uint32_t(5),
            ggml_backend_cuda_kv_stream_resident_pages_per_layer(runtime));

        ggml_backend_cuda_phase_arena_free(arena);
        std::fill(destination.begin(), destination.end(), 0);
        t.assert_true("KV runtime keeps the released arena alive",
            ggml_backend_cuda_kv_stream_stage_download(
                runtime, 1, 0, destination.data(), destination.size()));
        t.assert_true("retained arena bytes remain exact", source == destination);
        ggml_backend_cuda_kv_stream_runtime_free(runtime);
    });

    t.test("shared arena rejects overlap between KV and compute slices", [](testing & t) {
        constexpr size_t page_bytes = 64*1024;
        auto arena = ggml_backend_cuda_phase_arena_new(0, 20*page_bytes);
        if (!t.assert_true("arena allocation succeeds", arena != nullptr)) {
            return;
        }
        t.assert_true("compute range is accepted",
            ggml_backend_cuda_phase_arena_set_compute(
                arena, 10*page_bytes, 10*page_bytes));

        ggml_backend_cuda_kv_stream_params params{};
        params.device               = 0;
        params.stage_bytes          = page_bytes;
        params.stage_slots          = 2;
        params.pool_bytes           = 11*page_bytes;
        params.resident_layer_count = 2;
        params.page_tokens          = 256;
        t.assert_true("overlapping initial KV pool is rejected",
            ggml_backend_cuda_kv_stream_runtime_new_in_phase_arena(
                arena, 18*page_bytes, params) == nullptr);

        params.pool_bytes = 10*page_bytes;
        auto runtime = ggml_backend_cuda_kv_stream_runtime_new_in_phase_arena(
            arena, 18*page_bytes, params);
        if (!t.assert_true("non-overlapping KV runtime initializes", runtime != nullptr)) {
            ggml_backend_cuda_phase_arena_free(arena);
            return;
        }
        t.assert_true("compute cannot move into live KV",
            !ggml_backend_cuda_phase_arena_set_compute(
                arena, 9*page_bytes, 11*page_bytes));
        t.assert_true("KV cannot grow into compute",
            !ggml_backend_cuda_kv_stream_resize_pool(
                runtime, 11*page_bytes, 0, 2));
        t.assert_true("compute can move away from KV",
            ggml_backend_cuda_phase_arena_set_compute(
                arena, 12*page_bytes, 8*page_bytes));
        t.assert_true("KV can then grow to the new boundary",
            ggml_backend_cuda_kv_stream_resize_pool(
                runtime, 12*page_bytes, 0, 2));

        ggml_backend_cuda_kv_stream_runtime_free(runtime);
        ggml_backend_cuda_phase_arena_free(arena);
    });

    return t.summary();
}
