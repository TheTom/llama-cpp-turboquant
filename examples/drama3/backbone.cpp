#include "llama.h"
#include "llama-model.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: llama-drama3-backbone model.gguf [gpu_layers=0] [context=16384]\n";
        return 1;
    }
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    try {
        ggml_backend_load_all();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = argc > 2 ? std::stoi(argv[2]) : 0;
        model = llama_model_load_from_file(argv[1], mp);
        if (!model) {
            throw std::runtime_error("Cannot load backbone GGUF");
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = argc > 3 ? std::stoul(argv[3]) : 16384;
        cp.n_batch = 64;
        cp.n_ubatch = 8;
        cp.n_threads = 2;
        cp.n_threads_batch = 2;
        cp.embeddings = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            throw std::runtime_error("Cannot initialize backbone context");
        }
        const int n_embd = llama_model_n_embd(model);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const auto * embedding_tensor = model->tok_embd;
        if (!embedding_tensor || embedding_tensor->ne[0] != n_embd || embedding_tensor->ne[1] != n_vocab) {
            throw std::runtime_error("Invalid native token embedding table");
        }
        int position = 0;
        std::cout << json({{"ready", true}, {"n_embd", n_embd}, {"n_vocab", n_vocab}, {"embedding_type", ggml_type_name(embedding_tensor->type)}, {"scope", "backbone_only"}}).dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            auto request = json::parse(line);
            if (request.contains("token_ids")) {
                auto ids = request.at("token_ids").get<std::vector<int>>();
                if (ids.empty() || ids.size() > 64) {
                    throw std::runtime_error("Expected 1..64 token IDs");
                }
                const auto lookup_started = std::chrono::steady_clock::now();
                const size_t row_bytes = ggml_row_size(embedding_tensor->type, n_embd);
                std::vector<uint8_t> encoded(row_bytes);
                std::vector<std::vector<float>> embeddings;
                const auto * traits = ggml_get_type_traits(embedding_tensor->type);
                for (auto id : ids) {
                    if (id < 0 || id >= n_vocab) {
                        throw std::runtime_error("Token ID outside vocabulary");
                    }
                    ggml_backend_tensor_get(embedding_tensor, encoded.data(), id * embedding_tensor->nb[1], row_bytes);
                    embeddings.emplace_back(n_embd);
                    if (embedding_tensor->type == GGML_TYPE_F32) {
                        std::memcpy(embeddings.back().data(), encoded.data(), row_bytes);
                    } else if (traits->to_float) {
                        traits->to_float(encoded.data(), embeddings.back().data(), n_embd);
                    } else {
                        throw std::runtime_error("Unsupported native embedding type");
                    }
                }
                const double lookup_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - lookup_started).count();
                std::cout << json({{"token_embeddings", embeddings}, {"timings", {{"embedding_lookup_seconds", lookup_seconds}}}}).dump() << std::endl;
                continue;
            }
            if (request.value("reset", false)) {
                llama_memory_clear(llama_get_memory(ctx), true);
                position = 0;
                std::cout << json({{"reset", true}}).dump() << std::endl;
                continue;
            }
            auto rows = request.at("embeddings").get<std::vector<std::vector<float>>>();
            auto ids = request.at("candidate_ids").get<std::vector<int>>();
            if (rows.empty() || rows.size() > 64 || position + rows.size() > cp.n_ctx) {
                throw std::runtime_error("Expected 1..64 embedding rows within context");
            }
            for (auto id : ids) {
                if (id < 0 || id >= n_vocab) {
                    throw std::runtime_error("Candidate ID outside vocabulary");
                }
            }
            auto batch = llama_batch_init(rows.size(), n_embd, 1);
            batch.n_tokens = rows.size();
            for (size_t row = 0; row < rows.size(); ++row) {
                if (rows[row].size() != size_t(n_embd)) {
                    llama_batch_free(batch);
                    throw std::runtime_error("Embedding width does not match backbone");
                }
                for (int col = 0; col < n_embd; ++col) {
                    if (!std::isfinite(rows[row][col])) {
                        llama_batch_free(batch);
                        throw std::runtime_error("Nonfinite input embedding");
                    }
                    batch.embd[row * n_embd + col] = rows[row][col];
                }
                batch.pos[row] = position + row;
                batch.n_seq_id[row] = 1;
                batch.seq_id[row][0] = 0;
                batch.logits[row] = 1;
            }
            const auto decode_started = std::chrono::steady_clock::now();
            const int status = llama_decode(ctx, batch);
            llama_synchronize(ctx);
            const double decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_started).count();
            llama_batch_free(batch);
            if (status != 0) {
                throw std::runtime_error("Backbone decode failed: " + std::to_string(status));
            }
            const auto gather_started = std::chrono::steady_clock::now();
            std::vector<std::vector<float>> hidden, logits;
            for (size_t row = 0; row < rows.size(); ++row) {
                const auto * h = llama_get_embeddings_ith(ctx, row);
                const auto * l = llama_get_logits_ith(ctx, row);
                if (!h || !l) {
                    throw std::runtime_error("Backbone omitted hidden states or logits");
                }
                hidden.emplace_back(h, h + n_embd);
                logits.emplace_back();
                for (auto id : ids) {
                    logits.back().push_back(l[id]);
                }
            }
            const double gather_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - gather_started).count();
            std::cout << json({{"timings", {{"decode_synchronized_seconds", decode_seconds}, {"output_gather_seconds", gather_seconds}}}, {"position", position}, {"hidden_states", hidden}, {"candidate_ids", ids}, {"candidate_logits", logits}}).dump() << std::endl;
            position += rows.size();
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
