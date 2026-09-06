#include "kv-stream-bench-config.h"

#include <charconv>
#include <limits>
#include <string_view>

namespace {

constexpr uint64_t MIB = 1024ULL*1024ULL;

bool parse_u64(std::string_view text, uint64_t & value) {
    if (text.empty()) {
        return false;
    }

    const char * begin = text.data();
    const char * end   = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    return parsed.ec == std::errc() && parsed.ptr == end;
}

bool mib_to_bytes(uint64_t mib, uint64_t & bytes) {
    if (mib > std::numeric_limits<uint64_t>::max()/MIB) {
        return false;
    }

    bytes = mib*MIB;
    return true;
}

} // namespace

llama_kv_stream_bench_config llama_kv_stream_bench_config_parse(int argc, const char * const * argv) {
    llama_kv_stream_bench_config result;
    result.transfer_bytes = { 26*MIB, 197*MIB, 312*MIB, 416*MIB };

    auto fail = [&](const std::string & message) {
        result.valid = false;
        result.error = message;
        return result;
    };

    auto next_value = [&](int & index, std::string_view option, std::string_view & value) {
        if (index + 1 >= argc || argv[index + 1] == nullptr) {
            result.error = std::string(option) + " requires a value";
            return false;
        }

        value = argv[++index];
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        if (argv[i] == nullptr) {
            return fail("benchmark option must not be null");
        }

        const std::string_view option(argv[i]);
        if (option == "--execute") {
            result.execute = true;
            continue;
        }

        std::string_view value;
        if (option == "--device") {
            if (!next_value(i, option, value)) {
                return fail(result.error);
            }

            uint64_t parsed = 0;
            if (!parse_u64(value, parsed) || parsed > uint64_t(std::numeric_limits<int>::max())) {
                return fail("--device must be a non-negative integer");
            }
            result.device = int(parsed);
            continue;
        }

        if (option == "--iterations") {
            if (!next_value(i, option, value)) {
                return fail(result.error);
            }

            uint64_t parsed = 0;
            if (!parse_u64(value, parsed) || parsed == 0 || parsed > uint64_t(std::numeric_limits<int>::max())) {
                return fail("--iterations must be a positive integer");
            }
            result.iterations = int(parsed);
            continue;
        }

        if (option == "--reserve-mib") {
            if (!next_value(i, option, value)) {
                return fail(result.error);
            }

            uint64_t parsed = 0;
            if (!parse_u64(value, parsed) || !mib_to_bytes(parsed, result.reserve_bytes)) {
                return fail("--reserve-mib must be a valid non-negative integer");
            }
            continue;
        }

        if (option == "--sizes-mib") {
            if (!next_value(i, option, value)) {
                return fail(result.error);
            }

            std::vector<uint64_t> transfer_bytes;
            size_t begin = 0;
            while (begin <= value.size()) {
                const size_t comma = value.find(',', begin);
                const size_t end = comma == std::string_view::npos ? value.size() : comma;

                uint64_t parsed = 0;
                uint64_t bytes  = 0;
                if (!parse_u64(value.substr(begin, end - begin), parsed) ||
                    parsed == 0 || !mib_to_bytes(parsed, bytes)) {
                    return fail("--sizes-mib must be a comma-separated list of positive integers");
                }
                transfer_bytes.push_back(bytes);

                if (comma == std::string_view::npos) {
                    break;
                }
                begin = comma + 1;
            }

            result.transfer_bytes = std::move(transfer_bytes);
            continue;
        }

        return fail("unknown benchmark option: " + std::string(option));
    }

    result.valid = true;
    return result;
}
