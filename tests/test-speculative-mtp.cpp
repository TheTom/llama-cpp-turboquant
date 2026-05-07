#include "llama.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// Smoke tests for Gemma 4 MTP (aux head loaded into target).
// Set env vars to run non-skip paths; otherwise exits 0.

int main() {
    const char * path_tgt  = std::getenv("LLAMA_MTP_TEST_TARGET");
    const char * path_head = std::getenv("LLAMA_MTP_TEST_HEAD");
    const char * path_bad  = std::getenv("LLAMA_MTP_TEST_BAD_ARCH");
    const char * path_tgt_edge  = std::getenv("LLAMA_MTP_TEST_TARGET_EDGE");
    const char * path_head_edge = std::getenv("LLAMA_MTP_TEST_HEAD_EDGE");

    const bool run_primary = path_tgt && std::strlen(path_tgt) > 0;
    const bool run_edge    = path_tgt_edge && std::strlen(path_tgt_edge) > 0 && path_head_edge
        && std::strlen(path_head_edge) > 0;

    if (!run_primary && !run_edge) {
        std::cout << "skip: set LLAMA_MTP_TEST_TARGET (optional LLAMA_MTP_TEST_HEAD, LLAMA_MTP_TEST_BAD_ARCH), "
                     "and/or LLAMA_MTP_TEST_TARGET_EDGE + LLAMA_MTP_TEST_HEAD_EDGE for E2B/E4B centroid assistants\n";
        return 0;
    }

    llama_model_params mparams = llama_model_default_params();

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 512;
    cparams.n_batch     = 512;
    cparams.embeddings  = true;

    if (run_primary) {
    llama_model * model_tgt = llama_model_load_from_file(path_tgt, mparams);
    if (!model_tgt) {
        std::cerr << "failed to load target model\n";
        return 1;
    }

    if (std::strcmp(llama_model_arch_str(model_tgt), "gemma4") != 0) {
        std::cerr << "target arch must be gemma4\n";
        llama_model_free(model_tgt);
        return 1;
    }

    if (path_head && std::strlen(path_head) > 0) {
        if (llama_model_load_mtp_from_file(model_tgt, path_head, mparams) != 0) {
            std::cerr << "llama_model_load_mtp_from_file failed\n";
            llama_model_free(model_tgt);
            return 1;
        }
        if (!llama_model_has_mtp_assistant(model_tgt)) {
            std::cerr << "expected has_mtp_assistant after load\n";
            llama_model_free(model_tgt);
            return 1;
        }
        if (llama_model_get_mtp_assistant(model_tgt) == nullptr) {
            std::cerr << "expected get_mtp_assistant non-null\n";
            llama_model_free(model_tgt);
            return 1;
        }
        if (llama_model_mtp_n_embd_backbone(model_tgt) == 0) {
            std::cerr << "expected positive mtp n_embd_backbone\n";
            llama_model_free(model_tgt);
            return 1;
        }
    }

    llama_context * ctx = llama_init_from_model(model_tgt, cparams);
    if (!ctx) {
        std::cerr << "failed to create context\n";
        llama_model_free(model_tgt);
        return 1;
    }

    if (path_head && std::strlen(path_head) > 0) {
        // One forward pass so KV and last hidden exist; MTP step may still fail on bad fixtures — best-effort.
        llama_token bos = llama_vocab_bos(llama_model_get_vocab(model_tgt));
        if (bos == LLAMA_TOKEN_NULL) {
            bos = 0;
        }
        llama_batch b = llama_batch_get_one(&bos, 1);
        if (llama_decode(ctx, b) != 0) {
            std::cerr << "initial decode failed\n";
            llama_free(ctx);
            llama_model_free(model_tgt);
            return 1;
        }

        const uint32_t n_bb = llama_model_mtp_n_embd_backbone(model_tgt);
        std::vector<float> h_prev(n_bb, 0.f);
        if (float * h = llama_get_embeddings_ith(ctx, -1)) {
            const int no = llama_model_n_embd_out(model_tgt);
            const int nc = (int) std::min((size_t) no, (size_t) n_bb);
            std::memcpy(h_prev.data(), h, (size_t) nc * sizeof(float));
        }

        llama_memory_t mem = llama_get_memory(ctx);
        llama_pos attn_pos = mem ? llama_memory_seq_pos_max(mem, 0) : 0;
        if (attn_pos < 0) {
            attn_pos = 0;
        }

        llama_token drafts[4] = {};
        const int32_t rc = llama_decode_mtp(ctx, 0, attn_pos, bos, h_prev.data(), 1, drafts, nullptr, nullptr);
        if (rc != 0) {
            std::cerr << "llama_decode_mtp returned " << rc << " (fixture may be incomplete)\n";
            llama_free(ctx);
            llama_model_free(model_tgt);
            return 1;
        }

        // Async pipeline parity test (Phase E of async-mtp-pipeline plan):
        // sync facade and async submit/wait must agree on drafts for identical inputs.
        {
            llama_token drafts_async[4] = {};
            std::vector<float> h_async = h_prev;
            const int32_t rc_a = llama_decode_mtp_async(ctx, 0, attn_pos, bos, h_async.data(), 1);
            if (rc_a != 0) {
                std::cerr << "llama_decode_mtp_async returned " << rc_a << "\n";
                llama_free(ctx);
                llama_model_free(model_tgt);
                return 1;
            }
            const int32_t rc_w = llama_decode_mtp_wait(ctx, drafts_async, nullptr);
            if (rc_w != 0) {
                std::cerr << "llama_decode_mtp_wait returned " << rc_w << "\n";
                llama_free(ctx);
                llama_model_free(model_tgt);
                return 1;
            }
            if (drafts_async[0] != drafts[0]) {
                std::cerr << "async/sync MTP draft mismatch: sync=" << drafts[0]
                          << " async=" << drafts_async[0] << "\n";
                llama_free(ctx);
                llama_model_free(model_tgt);
                return 1;
            }
        }

        // Double-submit must be rejected without corrupting state.
        {
            std::vector<float> h2 = h_prev;
            if (llama_decode_mtp_async(ctx, 0, attn_pos, bos, h2.data(), 1) != 0) {
                std::cerr << "first async submit should succeed\n";
                llama_free(ctx);
                llama_model_free(model_tgt);
                return 1;
            }
            if (llama_decode_mtp_async(ctx, 0, attn_pos, bos, h2.data(), 1) != -7) {
                std::cerr << "second async submit before wait should return -7\n";
                llama_free(ctx);
                llama_model_free(model_tgt);
                return 1;
            }
            llama_token drain[4] = {};
            if (llama_decode_mtp_wait(ctx, drain, nullptr) != 0) {
                std::cerr << "drain wait should succeed\n";
                llama_free(ctx);
                llama_model_free(model_tgt);
                return 1;
            }
        }

        // Wait without prior submit must report "no in-flight" (-7).
        {
            llama_token drain[4] = {};
            if (llama_decode_mtp_wait(ctx, drain, nullptr) != -7) {
                std::cerr << "wait without submit should return -7\n";
                llama_free(ctx);
                llama_model_free(model_tgt);
                return 1;
            }
        }
    }

    llama_free(ctx);
    llama_model_free(model_tgt);
    }

    // Optional fixture: Gemma 4 Edge target + assistant with use_ordered_embeddings (centroid LM head).
    if (run_edge) {
        llama_model_params mparams_edge = llama_model_default_params();
        llama_model * model_edge = llama_model_load_from_file(path_tgt_edge, mparams_edge);
        if (!model_edge) {
            std::cerr << "failed to load LLAMA_MTP_TEST_TARGET_EDGE model\n";
            return 1;
        }
        if (std::strcmp(llama_model_arch_str(model_edge), "gemma4") != 0) {
            std::cerr << "EDGE target arch must be gemma4\n";
            llama_model_free(model_edge);
            return 1;
        }
        if (llama_model_load_mtp_from_file(model_edge, path_head_edge, mparams_edge) != 0) {
            std::cerr << "llama_model_load_mtp_from_file failed (EDGE head)\n";
            llama_model_free(model_edge);
            return 1;
        }
        llama_context * ctx_edge = llama_init_from_model(model_edge, cparams);
        if (!ctx_edge) {
            std::cerr << "failed to create EDGE context\n";
            llama_model_free(model_edge);
            return 1;
        }

        llama_token bos_e = llama_vocab_bos(llama_model_get_vocab(model_edge));
        if (bos_e == LLAMA_TOKEN_NULL) {
            bos_e = 0;
        }
        llama_batch b_e = llama_batch_get_one(&bos_e, 1);
        if (llama_decode(ctx_edge, b_e) != 0) {
            std::cerr << "EDGE initial decode failed\n";
            llama_free(ctx_edge);
            llama_model_free(model_edge);
            return 1;
        }

        const uint32_t n_bb_e = llama_model_mtp_n_embd_backbone(model_edge);
        std::vector<float> h_prev_e(n_bb_e, 0.f);
        if (float * h = llama_get_embeddings_ith(ctx_edge, -1)) {
            const int no = llama_model_n_embd_out(model_edge);
            const int nc = (int) std::min((size_t) no, (size_t) n_bb_e);
            std::memcpy(h_prev_e.data(), h, (size_t) nc * sizeof(float));
        }

        llama_memory_t mem_e = llama_get_memory(ctx_edge);
        llama_pos attn_pos_e = mem_e ? llama_memory_seq_pos_max(mem_e, 0) : 0;
        if (attn_pos_e < 0) {
            attn_pos_e = 0;
        }

        llama_token drafts_e[4] = {};
        if (llama_decode_mtp(ctx_edge, 0, attn_pos_e, bos_e, h_prev_e.data(), 1, drafts_e, nullptr, nullptr) != 0) {
            std::cerr << "EDGE llama_decode_mtp failed\n";
            llama_free(ctx_edge);
            llama_model_free(model_edge);
            return 1;
        }

        const int32_t n_vocab_e = llama_vocab_n_tokens(llama_model_get_vocab(model_edge));
        if (drafts_e[0] < 0 || drafts_e[0] >= n_vocab_e) {
            std::cerr << "EDGE MTP draft token out of vocabulary range\n";
            llama_free(ctx_edge);
            llama_model_free(model_edge);
            return 1;
        }

        // Async parity is enforced on the primary fixture; centroid MTP can diverge between schedulers (top_k / set_rows).

        llama_free(ctx_edge);
        llama_model_free(model_edge);
        std::cout << "mtp EDGE (E2B/E4B ordered embeddings) smoke ok\n";
    }

    if (run_primary && path_bad && std::strlen(path_bad) > 0) {
        llama_model * tgt2 = llama_model_load_from_file(path_tgt, mparams);
        if (!tgt2) {
            std::cerr << "failed to reload target for bad-arch test\n";
            return 1;
        }
        const int err = llama_model_load_mtp_from_file(tgt2, path_bad, mparams);
        llama_model_free(tgt2);
        if (err == 0) {
            std::cerr << "expected load_mtp to fail on incompatible GGUF\n";
            return 1;
        }
    }

    std::cout << "mtp aux-head smoke ok\n";
    return 0;
}
