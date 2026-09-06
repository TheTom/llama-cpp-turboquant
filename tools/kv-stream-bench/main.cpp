#include "kv-stream-bench-config.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

constexpr double MIB = 1024.0*1024.0;
constexpr double GIB = 1024.0*1024.0*1024.0;

bool cuda_ok(cudaError_t status, const char * operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    return false;
}

cudaError_t prefetch_async(void * pointer, size_t bytes, int device_id, cudaStream_t stream) {
#if CUDART_VERSION >= 13000
    cudaMemLocation location{};
    location.type = device_id == cudaCpuDeviceId ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
    location.id   = device_id == cudaCpuDeviceId ? 0 : device_id;
    return cudaMemPrefetchAsync(pointer, bytes, location, 0, stream);
#else
    return cudaMemPrefetchAsync(pointer, bytes, device_id, stream);
#endif
}

bool has_budget(uint64_t transfer_bytes, uint64_t reserve_bytes) {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo")) {
        return false;
    }

    const bool fits = transfer_bytes <= free_bytes && reserve_bytes <= free_bytes - transfer_bytes;
    if (!fits) {
        std::cout << "skip " << transfer_bytes/MIB << " MiB: free=" << free_bytes/MIB
                  << " MiB reserve=" << reserve_bytes/MIB << " MiB\n";
    }
    return fits;
}

void print_rate(const char * name, uint64_t bytes, int iterations, float milliseconds) {
    const double seconds = milliseconds/1000.0;
    const double gib_per_second = (double(bytes)*iterations/GIB)/seconds;

    std::cout << std::fixed << std::setprecision(2)
              << name << " size=" << bytes/MIB << " MiB"
              << " iterations=" << iterations
              << " elapsed=" << milliseconds << " ms"
              << " bandwidth=" << gib_per_second << " GiB/s\n";
}

bool benchmark_pinned(uint64_t bytes, int iterations) {
    void * host   = nullptr;
    void * device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end   = nullptr;

    bool ok = cuda_ok(cudaMallocHost(&host, size_t(bytes)), "cudaMallocHost") &&
              cuda_ok(cudaMalloc(&device, size_t(bytes)), "cudaMalloc") &&
              cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags") &&
              cuda_ok(cudaEventCreate(&begin), "cudaEventCreate(begin)") &&
              cuda_ok(cudaEventCreate(&end), "cudaEventCreate(end)");

    if (ok) {
        std::memset(host, 0x5a, size_t(bytes));
        ok = cuda_ok(cudaMemcpyAsync(device, host, size_t(bytes), cudaMemcpyHostToDevice, stream), "warmup cudaMemcpyAsync") &&
             cuda_ok(cudaStreamSynchronize(stream), "warmup cudaStreamSynchronize") &&
             cuda_ok(cudaEventRecord(begin, stream), "cudaEventRecord(begin)");
    }

    for (int i = 0; ok && i < iterations; ++i) {
        ok = cuda_ok(cudaMemcpyAsync(device, host, size_t(bytes), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
    }

    float milliseconds = 0.0f;
    if (ok) {
        ok = cuda_ok(cudaEventRecord(end, stream), "cudaEventRecord(end)") &&
             cuda_ok(cudaEventSynchronize(end), "cudaEventSynchronize") &&
             cuda_ok(cudaEventElapsedTime(&milliseconds, begin, end), "cudaEventElapsedTime");
    }
    if (ok) {
        print_rate("pinned-h2d", bytes, iterations, milliseconds);
    }

    if (end)    cudaEventDestroy(end);
    if (begin)  cudaEventDestroy(begin);
    if (stream) cudaStreamDestroy(stream);
    if (device) cudaFree(device);
    if (host)   cudaFreeHost(host);
    return ok;
}

bool benchmark_managed(uint64_t bytes, int iterations, int device_id) {
    void * managed = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end   = nullptr;

    bool ok = cuda_ok(cudaMallocManaged(&managed, size_t(bytes)), "cudaMallocManaged") &&
              cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags") &&
              cuda_ok(cudaEventCreate(&begin), "cudaEventCreate(begin)") &&
              cuda_ok(cudaEventCreate(&end), "cudaEventCreate(end)");

    if (ok) {
        std::memset(managed, 0x6b, size_t(bytes));
    }

    double total_milliseconds = 0.0;
    for (int i = 0; ok && i < iterations; ++i) {
        ok = cuda_ok(prefetch_async(managed, size_t(bytes), cudaCpuDeviceId, stream), "prefetch to CPU") &&
             cuda_ok(cudaStreamSynchronize(stream), "CPU prefetch synchronize") &&
             cuda_ok(cudaEventRecord(begin, stream), "cudaEventRecord(begin)") &&
             cuda_ok(prefetch_async(managed, size_t(bytes), device_id, stream), "prefetch to GPU");

        float milliseconds = 0.0f;
        if (ok) {
            ok = cuda_ok(cudaEventRecord(end, stream), "cudaEventRecord(end)") &&
                 cuda_ok(cudaEventSynchronize(end), "cudaEventSynchronize") &&
                 cuda_ok(cudaEventElapsedTime(&milliseconds, begin, end), "cudaEventElapsedTime");
            total_milliseconds += milliseconds;
        }
    }

    if (ok) {
        print_rate("managed-prefetch-h2d", bytes, iterations, float(total_milliseconds));
    }

    if (end)    cudaEventDestroy(end);
    if (begin)  cudaEventDestroy(begin);
    if (stream) cudaStreamDestroy(stream);
    if (managed) cudaFree(managed);
    return ok;
}

void print_plan(const llama_kv_stream_bench_config & config) {
    std::cout << "KV stream CUDA transfer benchmark\n"
              << "  mode: " << (config.execute ? "execute" : "dry-run") << '\n'
              << "  device: " << config.device << '\n'
              << "  iterations: " << config.iterations << '\n'
              << "  free-VRAM reserve: " << config.reserve_bytes/MIB << " MiB\n"
              << "  transfer sizes:";
    for (uint64_t bytes : config.transfer_bytes) {
        std::cout << ' ' << bytes/MIB << " MiB";
    }
    std::cout << "\nUse --execute to allocate and measure.\n";
}

} // namespace

int main(int argc, char ** argv) {
    const auto config = llama_kv_stream_bench_config_parse(argc, argv);
    if (!config.valid) {
        std::cerr << config.error << '\n';
        return 2;
    }

    print_plan(config);
    if (!config.execute) {
        return 0;
    }

    int device_count = 0;
    if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") ||
        config.device >= device_count ||
        !cuda_ok(cudaSetDevice(config.device), "cudaSetDevice")) {
        return 3;
    }

    cudaDeviceProp properties{};
    if (!cuda_ok(cudaGetDeviceProperties(&properties, config.device), "cudaGetDeviceProperties")) {
        return 3;
    }
    std::cout << "GPU: " << properties.name << '\n';

    for (uint64_t bytes : config.transfer_bytes) {
        if (!has_budget(bytes, config.reserve_bytes)) {
            continue;
        }
        if (!benchmark_pinned(bytes, config.iterations)) {
            return 4;
        }

        if (!has_budget(bytes, config.reserve_bytes)) {
            continue;
        }
        if (!benchmark_managed(bytes, config.iterations, config.device)) {
            return 5;
        }
    }

    return 0;
}
