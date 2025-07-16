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

#include <bench.hpp>
#include <iomanip>
#include <rocm_wmma_gemm/gemm.hpp>

template<m_layout a_layout, m_layout b_layout, m_layout c_layout>
void run_benchmark(benchmark::State& state, size_t M, size_t N, size_t K, size_t batch_count)
{
    // Allocate memory on host using std::vector
    matrix<__hip_bfloat16, a_layout> h_A(M, K);
    matrix<__hip_bfloat16, b_layout> h_B(K, N);
    matrix<__hip_bfloat16, c_layout> h_C(M, N);
    matrix<__hip_bfloat16, c_layout> h_C_ref(M, N);

    // Initialize input matrices with random values
    init_matrix(h_A);
    init_matrix(h_B);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Allocate memory on device
    __hip_bfloat16 *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, h_A.size() * batch_count * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_B, h_B.size() * batch_count * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_C, h_C.size() * batch_count * sizeof(__hip_bfloat16)));

    // Copy data from host to device
    for(int i = 0; i < batch_count; ++i)
    {
        HIP_CHECK(hipMemcpy(d_A + i * h_A.size(),
                            h_A.data(),
                            h_A.size() * sizeof(__hip_bfloat16),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B + i * h_B.size(),
                            h_B.data(),
                            h_B.size() * sizeof(__hip_bfloat16),
                            hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());

    gpu_timer timer;

    // Warmup only
    for(int i = 0; i < 5; ++i)
    {
        rocm_wmma_gemm::gemm<static_cast<rocm_wmma_gemm::m_layout>(c_layout),
                             static_cast<rocm_wmma_gemm::m_layout>(a_layout),
                             static_cast<rocm_wmma_gemm::m_layout>(b_layout)>(d_C,
                                                                              d_A,
                                                                              d_B,
                                                                              M,
                                                                              N,
                                                                              K,
                                                                              batch_count,
                                                                              stream);
        HIP_CHECK(hipPeekAtLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());

    double total_tflops = 0.0;
    double min_time     = std::numeric_limits<double>::max();
    double max_time     = 0.0;
    double total_flops  = 2.0 * M * N * K * batch_count;

    for(auto _ : state)
    {
        timer.start(stream);
        rocm_wmma_gemm::gemm<static_cast<rocm_wmma_gemm::m_layout>(c_layout),
                             static_cast<rocm_wmma_gemm::m_layout>(a_layout),
                             static_cast<rocm_wmma_gemm::m_layout>(b_layout)>(d_C,
                                                                              d_A,
                                                                              d_B,
                                                                              M,
                                                                              N,
                                                                              K,
                                                                              batch_count,
                                                                              stream);
        HIP_CHECK(hipPeekAtLastError());
        double elapsed_time = timer.stop(stream);

        double seconds = elapsed_time / 1000.0;
        state.SetIterationTime(seconds);

        // Track min/max times
        min_time = std::min(min_time, elapsed_time);
        max_time = std::max(max_time, elapsed_time);

        double tflops = (total_flops / seconds) * 1e-12;
        total_tflops += tflops;
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Set counters for the custom reporter
    state.counters["avg_tflops"]  = total_tflops / state.iterations();
    state.counters["min_time_ms"] = min_time;
    state.counters["max_time_ms"] = max_time;

    state.SetBytesProcessed(state.iterations() * ((M * K) + (K * N) + (M * N))
                            * sizeof(__hip_bfloat16));

    HIP_CHECK(hipDeviceSynchronize());
    // Free device memory and destroy stream
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

#define CREATE_BENCHMARK(LA, LB, LC, M, N, K)                                                  \
    benchmark::RegisterBenchmark("{A:" #LA ",B:" #LB ",C:" #LC ",m:" #M ",n:" #N ",k:" #K "}", \
                                 run_benchmark<LA, LB, LC>,                                    \
                                 M,                                                            \
                                 N,                                                            \
                                 K,                                                            \
                                 batch_count)

#define BENCHMARK_SIZE(lA, lB, lC)                      \
    CREATE_BENCHMARK(lA, lB, lC, 1024, 1024, 1024),     \
        CREATE_BENCHMARK(lA, lB, lC, 2048, 2048, 2048), \
        CREATE_BENCHMARK(lA, lB, lC, 4096, 4096, 4096), \
        CREATE_BENCHMARK(lA, lB, lC, 8192, 8192, 8192)

int main(int argc, char* argv[])
{
    // Parse argv
    benchmark::Initialize(&argc, argv);
    int trials      = -1;
    int batch_count = 1;

    std::vector<benchmark::internal::Benchmark*> benchmarks
        = {BENCHMARK_SIZE(m_layout::col_major, m_layout::col_major, m_layout::col_major),
           BENCHMARK_SIZE(m_layout::row_major, m_layout::col_major, m_layout::col_major),
           BENCHMARK_SIZE(m_layout::col_major, m_layout::row_major, m_layout::col_major),
           BENCHMARK_SIZE(m_layout::row_major, m_layout::row_major, m_layout::col_major),
           BENCHMARK_SIZE(m_layout::col_major, m_layout::col_major, m_layout::row_major),
           BENCHMARK_SIZE(m_layout::row_major, m_layout::col_major, m_layout::row_major),
           BENCHMARK_SIZE(m_layout::col_major, m_layout::row_major, m_layout::row_major),
           BENCHMARK_SIZE(m_layout::row_major, m_layout::row_major, m_layout::row_major)};

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

    // Use custom reporter instead of default
    CustomReporter reporter;
    benchmark::RunSpecifiedBenchmarks(&reporter);

    return 0;
}
