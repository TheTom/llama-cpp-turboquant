#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

struct ggml_cuda_kv_stream_resident_cache;
struct ggml_cuda_kv_stream_transfer_ring;

struct ggml_cuda_kv_stream_resident_stats {
    uint64_t resident_hits = 0;
    uint64_t resident_misses = 0;
    uint64_t streamed_pages = 0;
    uint64_t host_to_device_bytes = 0;
    uint64_t resident_attention_spans = 0;
    uint64_t resident_pages_attended = 0;
    uint64_t streamed_attention_spans = 0;
    uint64_t streamed_pages_attended = 0;
    uint64_t mma_prefill_attention_spans = 0;
};

struct ggml_cuda_kv_stream_transfer_stats {
    uint64_t asynchronous_page_uploads = 0;
    uint64_t host_to_device_copy_commands = 0;
    uint64_t compute_stream_waits = 0;
    uint64_t stage_slot_reuses = 0;
    uint64_t cross_layer_prefetches = 0;
    uint64_t deadline_samples = 0;
    uint64_t deadline_misses = 0;
    uint32_t ring_peak_occupancy = 0;
};

ggml_cuda_kv_stream_resident_cache * ggml_cuda_kv_stream_resident_cache_new(
    void * pool_data, size_t pool_bytes, size_t scratch_bytes, size_t page_bytes,
    uint32_t layer_count, uint32_t page_tokens);
void ggml_cuda_kv_stream_resident_cache_free(ggml_cuda_kv_stream_resident_cache * cache);
void ggml_cuda_kv_stream_resident_cache_reset(ggml_cuda_kv_stream_resident_cache * cache);
bool ggml_cuda_kv_stream_resident_cache_reconfigure(
    ggml_cuda_kv_stream_resident_cache * cache, size_t scratch_bytes,
    uint32_t active_pages_per_layer);
bool ggml_cuda_kv_stream_resident_cache_resize(
    ggml_cuda_kv_stream_resident_cache * cache, size_t pool_bytes,
    size_t scratch_bytes, uint32_t active_pages_per_layer);
bool ggml_cuda_kv_stream_resident_cache_repartition(
    ggml_cuda_kv_stream_resident_cache * cache, size_t scratch_bytes);
bool ggml_cuda_kv_stream_resident_cache_set_decode_layout(
    ggml_cuda_kv_stream_resident_cache * cache, uint32_t active_pages_per_layer);
uint32_t ggml_cuda_kv_stream_resident_cache_pages_per_layer(
    const ggml_cuda_kv_stream_resident_cache * cache);
uint32_t ggml_cuda_kv_stream_resident_cache_decode_active_pages(
    const ggml_cuda_kv_stream_resident_cache * cache);
ggml_cuda_kv_stream_resident_stats ggml_cuda_kv_stream_resident_cache_get_stats(
    const ggml_cuda_kv_stream_resident_cache * cache);
void ggml_cuda_kv_stream_resident_cache_mark_dirty(
    ggml_cuda_kv_stream_resident_cache * cache,
    const ggml_tensor * target, const ggml_tensor * indices);
bool ggml_cuda_kv_stream_resident_cache_mark_dirty_rows(
    ggml_cuda_kv_stream_resident_cache * cache,
    const int64_t * rows, size_t count);
bool ggml_cuda_kv_stream_resident_cache_all_layers_fit(
    const ggml_cuda_kv_stream_resident_cache * cache,
    uint32_t active_pages);
bool ggml_cuda_kv_stream_resident_cache_get_mirror(
    ggml_cuda_kv_stream_resident_cache * cache,
    const ggml_tensor * target,
    void ** data);
void ggml_cuda_kv_stream_resident_cache_mark_mirrored(
    ggml_cuda_kv_stream_resident_cache * cache,
    const ggml_tensor * target);

ggml_cuda_kv_stream_transfer_ring * ggml_cuda_kv_stream_transfer_ring_new(
    void * pool_data, size_t page_bytes, uint32_t stage_slots,
    void * conversion_data, size_t conversion_bytes,
    uint32_t forced_decode_span_pages);
void ggml_cuda_kv_stream_transfer_ring_free(ggml_cuda_kv_stream_transfer_ring * ring);
bool ggml_cuda_kv_stream_transfer_ring_set_active_slots(
    ggml_cuda_kv_stream_transfer_ring * ring, uint32_t stage_slots);
void ggml_cuda_kv_stream_transfer_ring_set_conversion_data(
    ggml_cuda_kv_stream_transfer_ring * ring, void * conversion_data);
void ggml_cuda_kv_stream_transfer_ring_reset_span_tuner(ggml_cuda_kv_stream_transfer_ring * ring);
bool ggml_cuda_kv_stream_transfer_ring_observe_decode_latency(
    ggml_cuda_kv_stream_transfer_ring * ring, double elapsed_ms);
ggml_cuda_kv_stream_transfer_stats ggml_cuda_kv_stream_transfer_ring_get_stats(
    const ggml_cuda_kv_stream_transfer_ring * ring);
void ggml_cuda_kv_stream_graph_begin(ggml_cuda_kv_stream_transfer_ring * ring);
bool ggml_cuda_kv_stream_graph_add_attention(
    ggml_cuda_kv_stream_transfer_ring * ring,
    ggml_cuda_kv_stream_resident_cache * resident_cache,
    const ggml_tensor * dst);
void ggml_cuda_kv_stream_graph_finalize(
    ggml_cuda_kv_stream_transfer_ring * ring, cudaStream_t compute_stream);
void ggml_cuda_kv_stream_graph_end(
    ggml_cuda_kv_stream_transfer_ring * ring, cudaStream_t compute_stream);
double ggml_cuda_kv_stream_copy_engine_busy_ratio(
    ggml_cuda_kv_stream_transfer_ring * ring);
uint32_t ggml_cuda_kv_stream_last_ring_peak_occupancy(
    const ggml_cuda_kv_stream_transfer_ring * ring);

bool ggml_cuda_kv_stream_page_bytes(
    ggml_type type_k, ggml_type type_v,
    uint32_t head_dim_k, uint32_t head_dim_v, uint32_t head_count,
    uint32_t page_tokens, size_t * page_bytes);
bool ggml_cuda_kv_stream_workspace_bytes(
    ggml_type type_k, ggml_type type_v,
    uint32_t head_dim_k, uint32_t head_dim_v, uint32_t head_count,
    uint32_t page_tokens, size_t * workspace_bytes);

bool ggml_cuda_flash_attn_ext_streamed_supported(const ggml_tensor * dst, size_t stage_bytes);

void ggml_cuda_flash_attn_ext_streamed(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    ggml_cuda_kv_stream_transfer_ring * transfer_ring,
    ggml_cuda_kv_stream_resident_cache * resident_cache);
