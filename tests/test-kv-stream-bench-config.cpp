#include "kv-stream-bench-config.h"
#include "testing.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr uint64_t MIB = 1024ULL*1024ULL;

llama_kv_stream_bench_config parse(std::initializer_list<const char *> args) {
    std::vector<const char *> argv(args);
    return llama_kv_stream_bench_config_parse(int(argv.size()), argv.data());
}

} // namespace

int main() {
    testing t;

    t.test("defaults are dry-run and cover observed transfer sizes", [](testing & t) {
        const auto config = parse({ "llama-kv-stream-bench" });

        if (!t.assert_true("configuration is valid", config.valid)) {
            return;
        }

        t.assert_true("execution requires opt in", !config.execute);
        t.assert_equal(0, config.device);
        t.assert_equal(5, config.iterations);
        t.assert_equal(uint64_t(1024)*MIB, config.reserve_bytes);
        t.assert_true("default sizes are exact", config.transfer_bytes == std::vector<uint64_t>({
            26*MIB, 197*MIB, 312*MIB, 416*MIB,
        }));
    });

    t.test("explicit safe options are parsed", [](testing & t) {
        const auto config = parse({
            "llama-kv-stream-bench",
            "--execute",
            "--device", "1",
            "--iterations", "3",
            "--reserve-mib", "2048",
            "--sizes-mib", "4,26,416",
        });

        if (!t.assert_true("configuration is valid", config.valid)) {
            return;
        }

        t.assert_true("execution is enabled", config.execute);
        t.assert_equal(1, config.device);
        t.assert_equal(3, config.iterations);
        t.assert_equal(uint64_t(2048)*MIB, config.reserve_bytes);
        t.assert_true("explicit sizes are exact", config.transfer_bytes == std::vector<uint64_t>({
            4*MIB, 26*MIB, 416*MIB,
        }));
    });

    t.test("unsafe or malformed options are rejected", [](testing & t) {
        t.assert_true("zero iterations are rejected", !parse({
            "bench", "--iterations", "0",
        }).valid);
        t.assert_true("zero transfer is rejected", !parse({
            "bench", "--sizes-mib", "26,0,416",
        }).valid);
        t.assert_true("negative device is rejected", !parse({
            "bench", "--device", "-1",
        }).valid);
        t.assert_true("unknown option is rejected", !parse({
            "bench", "--surprise",
        }).valid);
        t.assert_true("missing value is rejected", !parse({
            "bench", "--reserve-mib",
        }).valid);
    });

    return t.summary();
}
