#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_kv_stream_bench_config {
    bool valid   = false;
    bool execute = false;

    std::string error;

    int device     = 0;
    int iterations = 5;

    uint64_t reserve_bytes = 1024ULL*1024ULL*1024ULL;
    std::vector<uint64_t> transfer_bytes;
};

llama_kv_stream_bench_config llama_kv_stream_bench_config_parse(int argc, const char * const * argv);
