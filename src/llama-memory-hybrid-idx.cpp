#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   gdn_replay_req,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, gdn_replay_req, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, /* kv_stream_stage_bytes */ 0, "idx_");
    }()) {}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: this repeats llama_memory_hybrid::init_batch because the indexer cache needs the
    //       slot infos of the attention cache, which the base context does not expose
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer cache is addressed by the cells of the attention cache, so it takes that
        // slot layout instead of finding its own; a separate layout can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }

    // [TAG_PLE_HISTORY] every sequence is gone, so no window is trusted any more
    ple_hist.clear();
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it
    // first and leave the other caches untouched if it does
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    ple_hist_rm(seq_id, p0, p1);

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }

    ple_hist_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }

    ple_hist_keep(seq_id);
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }

    ple_hist_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }

    ple_hist_div(seq_id, p0, p1);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

//
// [TAG_PLE_HISTORY] per-sequence PLE n-gram history
//
// The window is only usable while it is contiguous with the position the sequence decodes next,
// so each operation below either rewrites it exactly or invalidates it with next_pos = -1.
// An invalid window makes set_input pad with EOS, which is what a fresh sequence also gets.
//

llama_memory_hybrid_idx::ple_history & llama_memory_hybrid_idx::ple_hist_get(llama_seq_id seq_id) const {
    return ple_hist[seq_id];
}

// first position still remembered by h
static llama_pos ple_hist_beg(const llama_memory_hybrid_idx::ple_history & h) {
    return h.next_pos - (llama_pos) h.toks.size();
}

static void ple_hist_invalidate(llama_memory_hybrid_idx::ple_history & h) {
    h.next_pos = -1;
    h.toks.clear();
}

// drop everything at position >= p, so the sequence now ends just before p
static void ple_hist_truncate(llama_memory_hybrid_idx::ple_history & h, llama_pos p) {
    const llama_pos beg = ple_hist_beg(h);

    if (p <= beg) {
        h.toks.clear();
    } else if (p < h.next_pos) {
        h.toks.resize((size_t) (p - beg));
    }

    h.next_pos = p;
}

void llama_memory_hybrid_idx::ple_hist_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (seq_id < 0) {
        for (auto it = ple_hist.begin(); it != ple_hist.end(); ) {
            const llama_seq_id id = it->first;
            ++it;
            ple_hist_rm(id, p0, p1);
        }
        return;
    }

    auto it = ple_hist.find(seq_id);
    if (it == ple_hist.end() || it->second.next_pos < 0) {
        return;
    }
    auto & h = it->second;

    if (p0 <= 0 && p1 < 0) {
        // the whole sequence is gone
        ple_hist.erase(it);
        return;
    }

    if (p1 < 0) {
        // a rewind: the sequence ends at p0 and the remaining prefix is still contiguous
        if (p0 < h.next_pos) {
            ple_hist_truncate(h, p0);
        }
        return;
    }

    // a hole in the middle: seq_rm does not renumber what follows, so an overlapping
    // window is no longer a run of consecutive positions
    if (p1 > ple_hist_beg(h) && p0 < h.next_pos) {
        ple_hist_invalidate(h);
    }
}

void llama_memory_hybrid_idx::ple_hist_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    // whatever the destination had is replaced by the copied range, exactly as its cells are
    ple_hist.erase(seq_id_dst);

    auto it = ple_hist.find(seq_id_src);
    if (it == ple_hist.end() || it->second.next_pos < 0) {
        return;
    }

    ple_history h = it->second;

    if (p1 >= 0 && p1 < h.next_pos) {
        ple_hist_truncate(h, p1);
    }

    // positions below p0 were not copied, so for the destination they are before the
    // sequence start, which the hash already reads as EOS
    const llama_pos lo = p0 < 0 ? 0 : p0;
    if (lo > ple_hist_beg(h)) {
        const llama_pos drop = std::min<llama_pos>(lo - ple_hist_beg(h), (llama_pos) h.toks.size());
        h.toks.erase(h.toks.begin(), h.toks.begin() + (size_t) drop);
    }

    ple_hist[seq_id_dst] = std::move(h);
}

void llama_memory_hybrid_idx::ple_hist_keep(llama_seq_id seq_id) {
    for (auto it = ple_hist.begin(); it != ple_hist.end(); ) {
        it = it->first == seq_id ? std::next(it) : ple_hist.erase(it);
    }
}

void llama_memory_hybrid_idx::ple_hist_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (seq_id < 0) {
        for (auto & it : ple_hist) {
            ple_hist_add(it.first, p0, p1, shift);
        }
        return;
    }

    auto it = ple_hist.find(seq_id);
    if (it == ple_hist.end() || it->second.next_pos < 0) {
        return;
    }
    auto & h = it->second;

    const llama_pos beg = ple_hist_beg(h);
    const llama_pos lo  = p0 < 0 ? 0 : p0;

    if (p1 >= 0 && p1 <= beg) {
        // entirely below the window: the tokens we remember keep their positions
        return;
    }
    if (lo >= h.next_pos) {
        // entirely above the window: nothing we remember moves
        return;
    }
    if (lo <= beg && (p1 < 0 || p1 >= h.next_pos)) {
        // the context-shift case: the whole window moves as one and stays consecutive
        if (beg + shift < 0) {
            ple_hist_invalidate(h);
        } else {
            h.next_pos += shift;
        }
        return;
    }

    // the shift cuts through the window and breaks its contiguity
    ple_hist_invalidate(h);
}

void llama_memory_hybrid_idx::ple_hist_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (seq_id < 0) {
        for (auto & it : ple_hist) {
            ple_hist_div(it.first, p0, p1);
        }
        return;
    }

    auto it = ple_hist.find(seq_id);
    if (it == ple_hist.end() || it->second.next_pos < 0) {
        return;
    }
    auto & h = it->second;

    const llama_pos lo = p0 < 0 ? 0 : p0;

    // division makes the positions non-consecutive, so any overlap ends the window
    if ((p1 < 0 || p1 > ple_hist_beg(h)) && lo < h.next_pos) {
        ple_hist_invalidate(h);
    }
}

// Serialised as a self-delimiting list so that seq_id == -1 (whole context) and a single
// sequence share one format:
//   u32 n_entries
//   n_entries * { i32 seq_id, i32 next_pos, u32 n_toks, i32 toks[n_toks] }
// n_toks is at most ple_ngram_size - 1, so an entry is a handful of bytes.
void llama_memory_hybrid_idx::ple_hist_state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    uint32_t n_entries = 0;
    for (const auto & it : ple_hist) {
        if ((seq_id < 0 || it.first == seq_id) && it.second.next_pos >= 0) {
            ++n_entries;
        }
    }

    io.write(&n_entries, sizeof(n_entries));

    for (const auto & it : ple_hist) {
        if ((seq_id >= 0 && it.first != seq_id) || it.second.next_pos < 0) {
            continue;
        }

        const int32_t  id       = it.first;
        const int32_t  next_pos = it.second.next_pos;
        const uint32_t n_toks   = (uint32_t) it.second.toks.size();

        io.write(&id,       sizeof(id));
        io.write(&next_pos, sizeof(next_pos));
        io.write(&n_toks,   sizeof(n_toks));
        if (n_toks > 0) {
            io.write(it.second.toks.data(), n_toks*sizeof(llama_token));
        }
    }
}

void llama_memory_hybrid_idx::ple_hist_state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    uint32_t n_entries = 0;
    io.read(&n_entries, sizeof(n_entries));

    // a single-sequence restore replaces one window, a whole-context one replaces them all,
    // as the caches around it do
    if (seq_id >= 0) {
        ple_hist.erase(seq_id);
    } else {
        ple_hist.clear();
    }

    for (uint32_t i = 0; i < n_entries; ++i) {
        int32_t  id       = 0;
        int32_t  next_pos = 0;
        uint32_t n_toks   = 0;

        io.read(&id,       sizeof(id));
        io.read(&next_pos, sizeof(next_pos));
        io.read(&n_toks,   sizeof(n_toks));

        // the window is never longer than ple_ngram_size - 1, so a larger count is a corrupt
        // blob and would size an allocation from the file
        if (n_toks > LLAMA_MAX_PLE_NGRAM - 1) {
            throw std::runtime_error("qwen4exp PLE history: implausible token count in state blob");
        }

        std::vector<llama_token> toks(n_toks);
        if (n_toks > 0) {
            io.read(toks.data(), n_toks*sizeof(llama_token));
        }

        // a single-sequence restore can target a different seq_id, so the destination wins
        const llama_seq_id dst = seq_id >= 0 ? seq_id : (llama_seq_id) id;

        auto & h = ple_hist[dst];
        h.next_pos = next_pos;
        h.toks     = std::move(toks);
    }
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section is written last, so it is a pure suffix of the
    // attn+recr layout: a reader that does not expect it stops early instead of misparsing it.
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

    // [TAG_PLE_HISTORY] last again, so this section is also a pure suffix.
    // It is not under the PARTIAL_ONLY gate: the window is recurrent state, the input the PLE
    // conv state comes from, and the recurrent cache is written for partial checkpoints too.
    ple_hist_state_write(io, seq_id);
}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: this repeats llama_memory_hybrid::state_read because the indexer cache has to be
    //       handed the cells the attention cache restored into, and because a restore that
    //       fails halfway has to leave all three caches in the same state

    // [TAG_HYBRID_IDX_SINFO]
    // The indexer cache is addressed by the cells of the attention cache, so its restore adopts
    // that layout instead of searching for cells of its own. Two independent find_slot calls
    // agree only while nothing makes the two caches see different occupancy, and a restore is
    // exactly the operation that can no longer promise that.
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

        // [TAG_PLE_HISTORY] must mirror the write order above
        ple_hist_state_read(io, seq_id);
    } catch (...) {
        // a half-restored context is the one state the indexer cache cannot be brought back from
        // by itself: the attention cache holds the restored cells and the indexer the old ones.
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly rather than through seq_rm, which the recurrent cache is allowed to
    // refuse and which would then clear the other two caches and not it
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }

    ple_hist.erase(seq_id);
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

llama_memory_hybrid_idx::ple_history & llama_memory_hybrid_idx_context::get_ple_hist(llama_seq_id seq_id) const {
    GGML_ASSERT(mem != nullptr);

    return mem->ple_hist_get(seq_id);
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = (float   *) bias->data;

    // block b covers [b*ratio, (b+1)*ratio), so its first token is at b*ratio
    // all mrope sections carry it: exact for text, approximate for images
    for (int64_t sec = 0; sec < 4; ++sec) {
        for (int64_t s = 0; s < n_ns; ++s) {
            for (int64_t b = 0; b < n_blocks; ++b) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = (int32_t) (b*r);
            }
        }
    }

    // one pass per stream: cell j is a different token in each, so no mapping is shared
    std::vector<int32_t> blk_of(n_kv);
    std::vector<int32_t> filled(n_blocks);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = mem->get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        // an incomplete block cannot be pooled; the bias below forces those tail cells in
        // -1 means no usable block, and block 0 only keeps the gather in range
        std::fill(blk_of.begin(),  blk_of.end(),  -1);
        std::fill(filled.begin(),  filled.end(),   0);
        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                continue;
            }

            const llama_pos p = cells.pos_get(j);
            const int64_t   b = p/r;

            if (b >= n_blocks) {
                continue;
            }

            blk_of[j] = (int32_t) b;
            cur_blk_cells[b*r + (p%r)] = (int32_t) j;
            filled[b]++;
        }

        for (int64_t j = 0; j < n_kv; ++j) {
            if (blk_of[j] >= 0 && filled[blk_of[j]] < r) {
                blk_of[j] = -1;
            }
            cur_cell_blk[j] = blk_of[j] < 0 ? 0 : blk_of[j];
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];
            const llama_pos    q      = ubatch->pos[i];

            // the tail is an incomplete block and is always visible, as in the reference
            const llama_pos tail_start = (q + 1)/r*r;

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id) && cells.pos_get(j) <= q) {
                    // finite, so it can never meet a -inf and produce a nan
                    v = cells.pos_get(j) >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                }

                cur_bias[j] = v;
            }
        }
    }
}
