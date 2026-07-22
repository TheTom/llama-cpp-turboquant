// SPDX-License-Identifier: MIT
// Build: g++ -O2 -std=c++17 -I/mnt/storage/Projects/turboquant/include \
//   -I/mnt/storage/Projects/turboquant/ggml/include \
//   kld_measure.cpp -o kld_measure \
//   -L/mnt/storage/Projects/turboquant/build-fixed-native/bin \
//   -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda \
//   -Wl,-rpath,/mnt/storage/Projects/turboquant/build-fixed-native/bin
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include "llama.h"

int main(int argc, char **argv) {
    std::string model_path;
    std::string prompt = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome.";
    int n_predict = 20;
    int ngl = 99;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i+1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i+1 < argc) n_predict = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ngl") && i+1 < argc) ngl = atoi(argv[++i]);
    }
    if (model_path.empty()) { fprintf(stderr, "Usage: ... -m model.gguf [-p prompt] [-n tokens]\n"); return 1; }

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    fprintf(stderr, "Loading model...\n");
    llama_model *model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    int n_vocab = llama_n_vocab(model);
    fprintf(stderr, "Vocabulary: %d\n", n_vocab);

    // Tokenize
    std::vector<llama_token> prompt_tokens(8192);
    int nt = llama_tokenize(model, prompt.data(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, false);
    prompt_tokens.resize(nt);
    fprintf(stderr, "Prompt tokens: %d\n", nt);

    // Test two cache types
    ggml_type cache_types[] = {GGML_TYPE_F16, GGML_TYPE_OSCAR2};
    const char *cache_names[] = {"f16", "oscar2"};
    std::vector<std::vector<float>> all_logits[2];

    for (int ci = 0; ci < 2; ci++) {
        fprintf(stderr, "\n--- Testing %s KV cache ---\n", cache_names[ci]);

        llama_cparams cparams = llama_context_default_params().cparams;
        cparams.n_ctx = 8192;
        cparams.n_batch = 2048;
        cparams.n_ubatch = 512;
        cparams.type_k = cache_types[ci];
        cparams.type_v = cache_types[ci];
        cparams.flash_attn = true;

        llama_context_params ctxp = llama_context_default_params();
        ctxp.cparams = cparams;

        llama_context *ctx = llama_new_context_with_model(model, ctxp);
        if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

        // Eval prompt
        if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size(), 0, 0))) {
            fprintf(stderr, "Failed to eval prompt\n"); return 1;
        }

        // Generate tokens, saving logits
        all_logits[ci].reserve(n_predict);
        std::vector<llama_token> output;
        for (int i = 0; i < n_predict; i++) {
            float *logits = llama_get_logits_ith(ctx, -1);
            if (!logits) { fprintf(stderr, "No logits at pos %d\n", i); break; }
            all_logits[ci].push_back(std::vector<float>(logits, logits + n_vocab));

            // Greedy sample
            llama_token id = (llama_token)std::distance(logits, std::max_element(logits, logits + n_vocab));
            output.push_back(id);

            llama_batch batch = llama_batch_get_one(&id, 1, prompt_tokens.size() + i, 0);
            if (llama_decode(ctx, batch)) { fprintf(stderr, "Failed at pos %d\n", i); break; }
        }

        // Print output
        std::string text;
        for (auto t : output) {
            char buf[16];
            int n = llama_token_to_piece(model, t, buf, sizeof(buf), 0, false);
            text.append(buf, std::max(0, n));
        }
        fprintf(stderr, "  Output: %s\n", text.c_str());

        llama_free(ctx);
    }

    // Compare
    fprintf(stderr, "\n=== KLD/Same-Top Results ===\n");
    int n_pos = std::min(all_logits[0].size(), all_logits[1].size());
    double total_kld = 0;
    int same_top = 0;

    for (int i = 0; i < (int)n_pos; i++) {
        const auto &l_f16 = all_logits[0][i];
        const auto &l_o2  = all_logits[1][i];

        // Softmax
        auto softmax = [&](const std::vector<float> &l) -> std::vector<float> {
            std::vector<float> p(n_vocab);
            float mx = *std::max_element(l.begin(), l.end());
            double sum = 0;
            for (int j = 0; j < n_vocab; j++) { p[j] = expf(l[j] - mx); sum += p[j]; }
            double inv = 1.0 / sum;
            for (int j = 0; j < n_vocab; j++) p[j] *= inv;
            return p;
        };

        auto p = softmax(l_f16);
        auto q = softmax(l_o2);

        double kl = 0;
        for (int j = 0; j < n_vocab; j++) {
            if (p[j] > 1e-15 && q[j] > 1e-15) kl += p[j] * log(p[j] / q[j]);
        }
        total_kld += kl;

        int top_f16 = (int)std::distance(l_f16.begin(), std::max_element(l_f16.begin(), l_f16.end()));
        int top_o2  = (int)std::distance(l_o2.begin(), std::max_element(l_o2.begin(), l_o2.end()));
        if (top_f16 == top_o2) same_top++;

        fprintf(stderr, "  pos %2d: KLD=%.8f  same-top=%s\n", i, kl, top_f16==top_o2?"Y":"N");
    }

    double mean_kld = total_kld / n_pos;
    fprintf(stderr, "\n  Mean KLD:     %.6f\n", mean_kld);
    fprintf(stderr, "  Same-top@1:  %.2f%% (%d/%d)\n", 100.0*same_top/n_pos, same_top, n_pos);

    // JSON output for machine parsing
    printf("{\"mean_kld\":%.6f,\"same_top_pct\":%.2f,\"positions\":%d}\n",
           mean_kld, 100.0*same_top/n_pos, n_pos);

    llama_free_model(model);
    llama_backend_free();
    return 0;
}
