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
#include <hip/hip_runtime.h>
#include <iostream>
#include <kernel/kernel.hpp>
#include <memory>
#include <random>
#include <vector>

using namespace rocm_wmma_gemm;

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

        std::mt19937                          gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for(size_t i = 0; i < M * K; ++i)
        {
            h_A[i] = half(dist(gen));
        }

        for(size_t i = 0; i < K * N; ++i)
        {
            h_B[i] = half(dist(gen));
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
            HIP_CHECK(hipFree(d_A));
        if(d_B)
            HIP_CHECK(hipFree(d_B));
        if(d_C)
            HIP_CHECK(hipFree(d_C));
        HIP_CHECK(hipStreamDestroy(stream));
    }
};

// Global test data instance
std::unique_ptr<test_data> g_test_data;

// Template function to run a specific kernel configuration
template<m_layout a_layout,
         m_layout b_layout,
         m_layout c_layout,
         int      warps_m,
         int      warps_n,
         int      warp_tile_m,
         int      warp_tile_n>
void run_kernel_config(benchmark::State& state)
{
    if(!g_test_data)
    {
        state.SkipWithError("Test data not initialized");
        return;
    }

    const size_t M = g_test_data->M;
    const size_t N = g_test_data->N;
    const size_t K = g_test_data->K;

    // Calculate grid dimensions
    constexpr int block_m = warps_m * warp_tile_m * wmma_tile;
    constexpr int block_n = warps_n * warp_tile_n * wmma_tile;

    const int grid_m       = (M + block_m - 1) / block_m;
    const int grid_n       = (N + block_n - 1) / block_n;
    const int total_blocks = grid_m * grid_n;

    dim3 grid_dim(total_blocks, 1);
    dim3 block_dim(warp_size * warps_m * warps_n);

    gpu_timer timer;

    // Warmup
    for(int i = 0; i < 5; ++i)
    {
        kernel_gemm<half, c_layout, a_layout, b_layout, warps_m, warps_n, warp_tile_m, warp_tile_n>
            <<<grid_dim, block_dim, 0, g_test_data->stream>>>(g_test_data->d_C,
                                                              g_test_data->d_A,
                                                              g_test_data->d_B,
                                                              M,
                                                              N,
                                                              K);
        HIP_CHECK(hipPeekAtLastError());
    }
    HIP_CHECK(hipStreamSynchronize(g_test_data->stream));

    // Benchmark loop
    for(auto _ : state)
    {
        timer.start(g_test_data->stream);
        kernel_gemm<half, c_layout, a_layout, b_layout, warps_m, warps_n, warp_tile_m, warp_tile_n>
            <<<grid_dim, block_dim, 0, g_test_data->stream>>>(g_test_data->d_C,
                                                              g_test_data->d_A,
                                                              g_test_data->d_B,
                                                              M,
                                                              N,
                                                              K);
        HIP_CHECK(hipPeekAtLastError());
        float elapsed_time = timer.stop(g_test_data->stream);
        HIP_CHECK(hipDeviceSynchronize());

        double seconds = elapsed_time / 1000.0;
        state.SetIterationTime(seconds);
    }
}

// Macro to register all layout combinations for a specific config
#define REGISTER_CONFIG(WM, WN, TM, TN)                                             \
    benchmark::RegisterBenchmark("config_" #WM "_" #WN "_" #TM "_" #TN "_rr_r",     \
                                 run_kernel_config<m_layout::row_major,             \
                                                   m_layout::row_major,             \
                                                   m_layout::row_major,             \
                                                   WM,                              \
                                                   WN,                              \
                                                   TM,                              \
                                                   TN>)                             \
        ->UseManualTime()                                                           \
        ->Unit(benchmark::kMillisecond),                                            \
        benchmark::RegisterBenchmark("config_" #WM "_" #WN "_" #TM "_" #TN "_rc_r", \
                                     run_kernel_config<m_layout::row_major,         \
                                                       m_layout::col_major,         \
                                                       m_layout::row_major,         \
                                                       WM,                          \
                                                       WN,                          \
                                                       TM,                          \
                                                       TN>)                         \
            ->UseManualTime()                                                       \
            ->Unit(benchmark::kMillisecond),                                        \
        benchmark::RegisterBenchmark("config_" #WM "_" #WN "_" #TM "_" #TN "_cr_r", \
                                     run_kernel_config<m_layout::col_major,         \
                                                       m_layout::row_major,         \
                                                       m_layout::row_major,         \
                                                       WM,                          \
                                                       WN,                          \
                                                       TM,                          \
                                                       TN>)                         \
            ->UseManualTime()                                                       \
            ->Unit(benchmark::kMillisecond),                                        \
        benchmark::RegisterBenchmark("config_" #WM "_" #WN "_" #TM "_" #TN "_cc_r", \
                                     run_kernel_config<m_layout::col_major,         \
                                                       m_layout::col_major,         \
                                                       m_layout::row_major,         \
                                                       WM,                          \
                                                       WN,                          \
                                                       TM,                          \
                                                       TN>)                         \
            ->UseManualTime()                                                       \
            ->Unit(benchmark::kMillisecond),                                        \
        benchmark::RegisterBenchmark("config_" #WM "_" #WN "_" #TM "_" #TN "_rr_c", \
                                     run_kernel_config<m_layout::row_major,         \
                                                       m_layout::row_major,         \
                                                       m_layout::col_major,         \
                                                       WM,                          \
                                                       WN,                          \
                                                       TM,                          \
                                                       TN>)                         \
            ->UseManualTime()                                                       \
            ->Unit(benchmark::kMillisecond),                                        \
        benchmark::RegisterBenchmark("config_" #WM "_" #WN "_" #TM "_" #TN "_rc_c", \
                                     run_kernel_config<m_layout::row_major,         \
                                                       m_layout::col_major,         \
                                                       m_layout::col_major,         \
                                                       WM,                          \
                                                       WN,                          \
                                                       TM,                          \
                                                       TN>)                         \
            ->UseManualTime()                                                       \
            ->Unit(benchmark::kMillisecond),                                        \
        benchmark::RegisterBenchmark("config_" #WM "_" #WN "_" #TM "_" #TN "_cr_c", \
                                     run_kernel_config<m_layout::col_major,         \
                                                       m_layout::row_major,         \
                                                       m_layout::col_major,         \
                                                       WM,                          \
                                                       WN,                          \
                                                       TM,                          \
                                                       TN>)                         \
            ->UseManualTime()                                                       \
            ->Unit(benchmark::kMillisecond),                                        \
        benchmark::RegisterBenchmark("config_" #WM "_" #WN "_" #TM "_" #TN "_cc_c", \
                                     run_kernel_config<m_layout::col_major,         \
                                                       m_layout::col_major,         \
                                                       m_layout::col_major,         \
                                                       WM,                          \
                                                       WN,                          \
                                                       TM,                          \
                                                       TN>)                         \
            ->UseManualTime()                                                       \
            ->Unit(benchmark::kMillisecond)

int main(int argc, char* argv[])
{
    if(argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " M N K [benchmark options...]" << std::endl;
        return 1;
    }

    size_t M = std::atol(argv[1]);
    size_t N = std::atol(argv[2]);
    size_t K = std::atol(argv[3]);

    int trials = -1;

    // Initialize global test data
    g_test_data = std::make_unique<test_data>(M, N, K);

    // Remove M, N, K arguments so benchmark can parse its own args
    argc -= 3;
    argv += 3;

    // Initialize benchmark with adjusted argc/argv
    benchmark::Initialize(&argc, argv);

    std::vector<benchmark::internal::Benchmark*> benchmarks = {
        REGISTER_CONFIG(2, 2, 2, 2), REGISTER_CONFIG(2, 2, 2, 4), REGISTER_CONFIG(2, 2, 2, 8),
        REGISTER_CONFIG(2, 2, 4, 2), REGISTER_CONFIG(2, 2, 4, 4), REGISTER_CONFIG(2, 2, 4, 8),
        REGISTER_CONFIG(2, 2, 8, 2), REGISTER_CONFIG(2, 2, 8, 4), REGISTER_CONFIG(2, 2, 8, 8),

        REGISTER_CONFIG(2, 4, 2, 2), REGISTER_CONFIG(2, 4, 2, 4), REGISTER_CONFIG(2, 4, 2, 8),
        REGISTER_CONFIG(2, 4, 4, 2), REGISTER_CONFIG(2, 4, 4, 4), REGISTER_CONFIG(2, 4, 4, 8),
        REGISTER_CONFIG(2, 4, 8, 2), REGISTER_CONFIG(2, 4, 8, 4),

        REGISTER_CONFIG(2, 8, 2, 2), REGISTER_CONFIG(2, 8, 2, 4), REGISTER_CONFIG(2, 8, 4, 2),
        REGISTER_CONFIG(2, 8, 4, 4), REGISTER_CONFIG(2, 8, 8, 2), REGISTER_CONFIG(2, 8, 8, 4),

        REGISTER_CONFIG(4, 2, 2, 2), REGISTER_CONFIG(4, 2, 2, 4), REGISTER_CONFIG(4, 2, 2, 8),
        REGISTER_CONFIG(4, 2, 4, 2), REGISTER_CONFIG(4, 2, 4, 4), REGISTER_CONFIG(4, 2, 4, 8),
        REGISTER_CONFIG(4, 2, 8, 2), REGISTER_CONFIG(4, 2, 8, 4), REGISTER_CONFIG(4, 2, 8, 8),

        REGISTER_CONFIG(4, 4, 2, 2), REGISTER_CONFIG(4, 4, 2, 4), REGISTER_CONFIG(4, 4, 2, 8),
        REGISTER_CONFIG(4, 4, 4, 2), REGISTER_CONFIG(4, 4, 4, 4), REGISTER_CONFIG(4, 4, 4, 8),
        REGISTER_CONFIG(4, 4, 8, 2), REGISTER_CONFIG(4, 4, 8, 4), REGISTER_CONFIG(4, 4, 8, 8),

        REGISTER_CONFIG(4, 8, 2, 2), REGISTER_CONFIG(4, 8, 2, 4), REGISTER_CONFIG(4, 8, 4, 2),
        REGISTER_CONFIG(4, 8, 4, 4), REGISTER_CONFIG(4, 8, 8, 2), REGISTER_CONFIG(4, 8, 8, 4),

        REGISTER_CONFIG(8, 2, 2, 2), REGISTER_CONFIG(8, 2, 2, 4), REGISTER_CONFIG(8, 2, 2, 8),
        REGISTER_CONFIG(8, 2, 4, 2), REGISTER_CONFIG(8, 2, 4, 4), REGISTER_CONFIG(8, 2, 4, 8),

        REGISTER_CONFIG(8, 4, 2, 2), REGISTER_CONFIG(8, 4, 2, 4), REGISTER_CONFIG(8, 4, 2, 8),
        REGISTER_CONFIG(8, 4, 4, 2), REGISTER_CONFIG(8, 4, 4, 4), REGISTER_CONFIG(8, 4, 4, 8),
    };

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();

    // Clean up global test data
    g_test_data.reset();

    return 0;
}
