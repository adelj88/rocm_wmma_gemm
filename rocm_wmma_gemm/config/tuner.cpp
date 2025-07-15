/*
 * MIT License
 *
 * Copyright (c) 2024 Adel Johar
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <benchmark/benchmark.h>
#include <fstream>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <iostream>
#include <memory>
#include <rocm_wmma_gemm/kernel/common.hpp>
#include <sstream>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                                        \
    {                                                                                          \
        hipError_t status = call;                                                              \
        if(status != hipSuccess)                                                               \
        {                                                                                      \
            std::cerr << "HIP error: " #call " failed with error " << static_cast<int>(status) \
                      << ": " << hipGetErrorString(status) << std::endl;                       \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    }

#define HIPRTC_CHECK(call)                                                                        \
    {                                                                                             \
        hiprtcResult status = call;                                                               \
        if(status != HIPRTC_SUCCESS)                                                              \
        {                                                                                         \
            std::cerr << "HIPRTC error: " #call " failed with error " << static_cast<int>(status) \
                      << std::endl;                                                               \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    }

class gpu_timer
{
public:
    gpu_timer()
    {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }

    ~gpu_timer()
    {
        HIP_CHECK(hipEventDestroy(start_));
        HIP_CHECK(hipEventDestroy(stop_));
    }

    void start(hipStream_t& stream)
    {
        HIP_CHECK(hipEventRecord(start_, stream));
    }

    float stop(hipStream_t& stream)
    {
        HIP_CHECK(hipEventRecord(stop_, stream));
        HIP_CHECK(hipEventSynchronize(stop_));
        float elapsed = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start_, stop_));
        return elapsed;
    }

private:
    hipEvent_t start_, stop_;
};

// Global test data structure
struct test_data
{
    half*       d_A = nullptr;
    half*       d_B = nullptr;
    half*       d_C = nullptr;
    hipStream_t stream;
    size_t      M, N, K;

    test_data(size_t m, size_t n, size_t k) : M(m), N(n), K(k)
    {
        // Initialize random data
        std::vector<half> h_A(M * K);
        std::vector<half> h_B(K * N);

        constexpr float  values[]   = {0.1f, 0.125f, 0.15f, 0.175f, 0.2f};
        constexpr size_t num_values = sizeof(values) / sizeof(values[0]);

        size_t idx = 0;
        for(size_t i = 0; i < M * K; ++i)
        {
            h_A[i] = half(values[idx % num_values]);
            idx++;
        }

        idx = 0;
        for(size_t i = 0; i < K * N; ++i)
        {
            h_B[i] = half(values[idx % num_values]);
            idx++;
        }

        // Create stream
        HIP_CHECK(hipStreamCreate(&stream));

        // Allocate device memory
        HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(half)));
        HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(half)));
        HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(half)));

        // Copy data to device
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), K * N * sizeof(half), hipMemcpyHostToDevice));

        HIP_CHECK(hipDeviceSynchronize());
    }

    ~test_data()
    {
        if(d_A)
        {
            HIP_CHECK(hipFree(d_A));
        }

        if(d_B)
        {
            HIP_CHECK(hipFree(d_B));
        }

        if(d_C)
        {
            HIP_CHECK(hipFree(d_C));
        }

        HIP_CHECK(hipStreamDestroy(stream));
    }
};

// Configuration parameters
struct config_params
{
    int         warps_m;
    int         warps_n;
    int         warp_tile_m;
    int         warp_tile_n;
    int         layout_a; // 0=row_major, 1=col_major
    int         layout_b; // 0=row_major, 1=col_major
    int         layout_c; // 0=row_major, 1=col_major
    std::string gpu_arch;
};

// Global test data and config
std::unique_ptr<test_data> g_test_data;
config_params              g_config;
hipModule_t                g_module      = nullptr;
hipFunction_t              g_kernel_func = nullptr;

// Generate kernel source code for specific configuration
std::string generate_kernel_source(const config_params& config)
{
    std::stringstream kernel_source;

    // Include the existing headers
    kernel_source << R"(
// Define missing std functions for hipRTC
namespace std
{
    template<class T>
    __device__ __host__ constexpr const T& min(const T& a, const T& b)
    {
        return (b < a) ? b : a;
    }

    template<class T>
    __device__ __host__ constexpr const T& max(const T& a, const T& b)
    {
        return (a < b) ? b : a;
    }
}

#include <kernel/kernel.hpp>

namespace rocm_wmma_gemm
{

// Extern template declaration - this IS the kernel we'll call
template __global__ void kernel_gemm<half, half, )"
                  << (config.layout_c == 0 ? "m_layout::row_major" : "m_layout::col_major") << ", "
                  << (config.layout_a == 0 ? "m_layout::row_major" : "m_layout::col_major") << ", "
                  << (config.layout_b == 0 ? "m_layout::row_major" : "m_layout::col_major") << ", "
                  << config.warps_m << ", " << config.warps_n << ", " << config.warp_tile_m << ", "
                  << config.warp_tile_n << R"(>(
    half* C, const half* A, const half* B, int M, int N, int K);

} // namespace rocm_wmma_gemm
)";

    return kernel_source.str();
}

// Read header file contents
std::string read_header_file(const std::string& filepath)
{
    std::ifstream file(filepath);
    if(!file.is_open())
    {
        std::cerr << "Failed to open header file: " << filepath << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Compile kernel using hipRTC
bool compile_kernel(const config_params& config)
{
    // Clean up previous module if exists
    if(g_module)
    {
        HIP_CHECK(hipModuleUnload(g_module));
        g_module      = nullptr;
        g_kernel_func = nullptr;
    }

    // Generate kernel source
    std::string kernel_source = generate_kernel_source(config);

    // Read required headers
    std::vector<std::string> header_sources;
    std::vector<const char*> header_names;
    std::vector<const char*> header_contents;

    // Add required headers - adjust paths as needed
    std::string include_dir = PROJECT_SOURCE_DIR "/include/rocm_wmma_gemm";

    std::vector<std::string> required_headers = {include_dir + "/kernel/kernel.hpp",
                                                 include_dir + "/kernel/common.hpp",
                                                 include_dir + "/kernel/fragment.hpp",
                                                 include_dir + "/kernel/load.hpp",
                                                 include_dir + "/kernel/mapping.hpp",
                                                 include_dir + "/kernel/wmma.hpp"};

    std::vector<std::string> header_name_strings = {"kernel/kernel.hpp",
                                                    "kernel/common.hpp",
                                                    "kernel/fragment.hpp",
                                                    "kernel/load.hpp",
                                                    "kernel/mapping.hpp",
                                                    "kernel/wmma.hpp"};

    // Read all header files
    for(size_t i = 0; i < required_headers.size(); ++i)
    {
        std::string content = read_header_file(required_headers[i]);
        if(content.empty())
        {
            std::cerr << "Failed to read header: " << required_headers[i] << std::endl;
            return false;
        }
        header_sources.push_back(content);
    }

    // Prepare header arrays for hipRTC
    for(size_t i = 0; i < header_sources.size(); ++i)
    {
        header_contents.push_back(header_sources[i].c_str());
        header_names.push_back(header_name_strings[i].c_str());
    }

    // Create hipRTC program with headers
    hiprtcProgram prog;
    HIPRTC_CHECK(hiprtcCreateProgram(&prog,
                                     kernel_source.c_str(),
                                     "dynamic_kernel.hip",
                                     header_contents.size(),
                                     header_contents.data(),
                                     header_names.data()));

    // Build the full template instantiation string for name expression
    std::stringstream kernel_name_ss;
    kernel_name_ss << "rocm_wmma_gemm::kernel_gemm<half, half, "
                   << (config.layout_c == 0 ? "rocm_wmma_gemm::m_layout::row_major"
                                            : "rocm_wmma_gemm::m_layout::col_major")
                   << ", "
                   << (config.layout_a == 0 ? "rocm_wmma_gemm::m_layout::row_major"
                                            : "rocm_wmma_gemm::m_layout::col_major")
                   << ", "
                   << (config.layout_b == 0 ? "rocm_wmma_gemm::m_layout::row_major"
                                            : "rocm_wmma_gemm::m_layout::col_major")
                   << ", " << config.warps_m << ", " << config.warps_n << ", " << config.warp_tile_m
                   << ", " << config.warp_tile_n << ">";

    std::string kernel_name = kernel_name_ss.str();
    std::cout << "Adding name expression: " << kernel_name << std::endl;

    // Add name expression for the specific template instantiation BEFORE compilation
    HIPRTC_CHECK(hiprtcAddNameExpression(prog, kernel_name.c_str()));

    // Set compilation options
    std::vector<const char*> options
        = {"-O3", "-ffast-math", "-mcumode", "-std=c++17", config.gpu_arch.c_str()};

    // Compile the program
    hiprtcResult compile_result = hiprtcCompileProgram(prog, options.size(), options.data());

    // Check for compilation errors
    if(compile_result != HIPRTC_SUCCESS)
    {
        size_t log_size;
        HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &log_size));

        if(log_size > 1)
        {
            std::vector<char> log(log_size);
            HIPRTC_CHECK(hiprtcGetProgramLog(prog, log.data()));
            std::cerr << "Compilation failed:\n" << log.data() << std::endl;
        }

        HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
        return false;
    }

    // Get the mangled name of the kernel AFTER compilation
    const char*  lowered_name_ptr = nullptr;
    hiprtcResult name_result = hiprtcGetLoweredName(prog, kernel_name.c_str(), &lowered_name_ptr);

    std::string mangled_name;
    if(name_result == HIPRTC_SUCCESS && lowered_name_ptr != nullptr)
    {
        mangled_name = std::string(lowered_name_ptr);
        std::cout << "Mangled kernel name: " << mangled_name << std::endl;
    }
    else
    {
        std::cerr << "Failed to get mangled name, trying original name" << std::endl;
        mangled_name = kernel_name; // Fallback to original name
    }

    // Get compiled code
    size_t code_size;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));

    std::vector<char> code(code_size);
    HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));

    // Clean up program BEFORE loading module
    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

    // Load module
    HIP_CHECK(hipModuleLoadData(&g_module, code.data()));

    // Get kernel function using the mangled name
    HIP_CHECK(hipModuleGetFunction(&g_kernel_func, g_module, mangled_name.c_str()));

    return true;
}

// Benchmark function for the dynamically compiled kernel
void run_kernel_benchmark(benchmark::State& state)
{
    if(!g_test_data || !g_kernel_func)
    {
        state.SkipWithError("Test data or kernel not initialized");
        return;
    }

    const size_t M = g_test_data->M;
    const size_t N = g_test_data->N;
    const size_t K = g_test_data->K;

    // Calculate grid dimensions
    const int block_m = g_config.warps_m * g_config.warp_tile_m * rocm_wmma_gemm::wmma_tile;
    const int block_n = g_config.warps_n * g_config.warp_tile_n * rocm_wmma_gemm::wmma_tile;

    const int grid_m       = (M + block_m - 1) / block_m;
    const int grid_n       = (N + block_n - 1) / block_n;
    const int total_blocks = grid_m * grid_n;

    dim3 grid_dim(total_blocks, 1);
    dim3 block_dim(rocm_wmma_gemm::warp_size * g_config.warps_m * g_config.warps_n);

    gpu_timer timer;

    void* args[] = {&g_test_data->d_C,
                    &g_test_data->d_A,
                    &g_test_data->d_B,
                    const_cast<int*>(reinterpret_cast<const int*>(&M)),
                    const_cast<int*>(reinterpret_cast<const int*>(&N)),
                    const_cast<int*>(reinterpret_cast<const int*>(&K))};

    // Warmup
    for(int i = 0; i < 5; ++i)
    {
        HIP_CHECK(hipModuleLaunchKernel(g_kernel_func,
                                        grid_dim.x,
                                        grid_dim.y,
                                        grid_dim.z,
                                        block_dim.x,
                                        block_dim.y,
                                        block_dim.z,
                                        0,
                                        g_test_data->stream,
                                        args,
                                        nullptr));
        HIP_CHECK(hipPeekAtLastError());
    }
    HIP_CHECK(hipStreamSynchronize(g_test_data->stream));

    // Benchmark loop
    for(auto _ : state)
    {
        timer.start(g_test_data->stream);

        HIP_CHECK(hipModuleLaunchKernel(g_kernel_func,
                                        grid_dim.x,
                                        grid_dim.y,
                                        grid_dim.z,
                                        block_dim.x,
                                        block_dim.y,
                                        block_dim.z,
                                        0,
                                        g_test_data->stream,
                                        args,
                                        nullptr));
        HIP_CHECK(hipPeekAtLastError());

        float elapsed_time = timer.stop(g_test_data->stream);
        HIP_CHECK(hipDeviceSynchronize());

        double seconds = elapsed_time / 1000.0;
        state.SetIterationTime(seconds);
    }
}

int main(int argc, char* argv[])
{
    if(argc != 12)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " M N K warps_m warps_n warp_tile_m warp_tile_n layout_a layout_b layout_c gpu_arch"
            << std::endl;
        std::cerr << "Example: " << argv[0] << " 4096 4096 4096 4 4 4 4 0 1 0 gfx1100" << std::endl;
        return 1;
    }

    // Parse command line arguments
    size_t M = std::atol(argv[1]);
    size_t N = std::atol(argv[2]);
    size_t K = std::atol(argv[3]);

    g_config.warps_m     = std::atoi(argv[4]);
    g_config.warps_n     = std::atoi(argv[5]);
    g_config.warp_tile_m = std::atoi(argv[6]);
    g_config.warp_tile_n = std::atoi(argv[7]);
    g_config.layout_a    = std::atoi(argv[8]);
    g_config.layout_b    = std::atoi(argv[9]);
    g_config.layout_c    = std::atoi(argv[10]);
    std::string tmp      = argv[11];
    g_config.gpu_arch    = "--offload-arch=" + tmp;

    // Initialize test data
    g_test_data = std::make_unique<test_data>(M, N, K);

    // Compile kernel with hipRTC
    if(!compile_kernel(g_config))
    {
        std::cerr << "Failed to compile kernel" << std::endl;
        return 1;
    }

    std::cout << "Successfully compiled kernel for config: " << g_config.warps_m << ","
              << g_config.warps_n << "," << g_config.warp_tile_m << "," << g_config.warp_tile_n
              << "," << g_config.layout_a << "," << g_config.layout_b << "," << g_config.layout_c
              << std::endl;

    // Initialize benchmark
    benchmark::Initialize(&argc, argv);

    // Register the benchmark
    benchmark::RegisterBenchmark("dynamic_kernel", run_kernel_benchmark)
        ->UseManualTime()
        ->Unit(benchmark::kMillisecond);

    // Run benchmark
    benchmark::RunSpecifiedBenchmarks();

    // Clean up
    if(g_module)
    {
        HIP_CHECK(hipModuleUnload(g_module));
    }
    g_test_data.reset();

    return 0;
}
