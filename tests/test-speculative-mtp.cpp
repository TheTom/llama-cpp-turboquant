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

    if (!path_tgt || std::strlen(path_tgt) == 0) {
        std::cout << "skip: set LLAMA_MTP_TEST_TARGET; optional LLAMA_MTP_TEST_HEAD, LLAMA_MTP_TEST_BAD_ARCH\n";
        return 0;
    }

    llama_model_params mparams = llama_model_default_params();
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

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 512;
    cparams.n_batch     = 512;
    cparams.embeddings  = true;

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
    }

    llama_free(ctx);
    llama_model_free(model_tgt);

    if (path_bad && std::strlen(path_bad) > 0) {
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
