#pragma once

#include "llama.h"
#include "llama-graph.h"

#include <map>
#include <memory>
#include <functional>
#include <vector>

struct llama_ubatch;

class llama_batch_allocr;

class llama_io_write_i;
class llama_io_read_i;

class llama_kv_cache;

// A sub-cache eligible for the shared CUDA phase-arena streaming runtime
// (block KV streaming - see llama-context.cpp's kv_stream_switch_phase and
// llama-kv-cache.cpp's kv_stream_runtime). Most memory types have none; a
// wrapper holding more than one llama_kv_cache (iSWA, DSA, MSA, DSV4)
// returns whichever of its sub-cache(s) actually got a streaming runtime
// attached - see llama_memory_i::get_kv_stream_targets().
struct llama_kv_stream_target {
    llama_kv_cache * cache = nullptr;
};

// The per-ubatch live state of one streaming target, used to drive the
// phase-switch/adapt feedback loop each step - see
// llama_memory_context_i::get_kv_stream_active_targets().
struct llama_kv_stream_active_target {
    llama_kv_cache * cache = nullptr;
    uint32_t n_kv = 0;
};

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;
    uint64_t kv_stream_stage_bytes;
    void * kv_stream_phase_arena;
    uint64_t kv_stream_maximum_pool_bytes;

    // use full-size SWA cache
    bool swa_full;

    llama_context_type ctx_type;

    llama_memory_t mem_other;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llama_memory_status_is_fail(llama_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;

    // TurboQuant: get rotation tensors for pre-rotate-queries optimization
    // Returns null for non-turbo memory types. Override in KV cache contexts.
    virtual ggml_tensor * get_turbo_rot_forward() const { return nullptr; }
    virtual ggml_tensor * get_turbo_rot_inverse() const { return nullptr; }

    // TurboQuant InnerQ: get per-channel scale_inv tensor for Q/V equalization
    // Returns nullptr when InnerQ is not active. Override in KV cache contexts.
    virtual ggml_tensor * get_turbo_innerq_scale_inv() const { return nullptr; }

    // Block KV streaming: this ubatch's live state for whichever of this
    // memory's sub-cache(s) have a streaming runtime attached. Empty for
    // every memory type that doesn't stream (the default), or hasn't been
    // wired up to yet. Drives the per-ubatch phase-switch/adapt call in
    // llama_context::process_ubatch - see llama-context.cpp.
    virtual std::vector<llama_kv_stream_active_target> get_kv_stream_active_targets() const { return {}; }
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    using layer_share_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;

    // Block KV streaming: the sub-cache(s) of this memory object, if any,
    // that a CUDA phase-arena streaming runtime should be attached to.
    // Empty for every memory type that doesn't support streaming (the
    // default). A plain llama_kv_cache returns itself once it has a
    // runtime; a multi-cache wrapper (iSWA, DSA, MSA, DSV4) returns
    // whichever of its sub-cache(s) are the large, context-length-scaled
    // ones worth streaming - narrow/bounded indexer or compression-state
    // structures stay always-resident and are never returned here. See
    // llama-context.cpp's kv_stream_switch_phase and the bootstrap pre-scan.
    virtual std::vector<llama_kv_stream_target> get_kv_stream_targets() const { return {}; }
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
