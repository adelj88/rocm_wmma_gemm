#include <array>
#include <chrono>
#include <gemm.hpp>
#include <hip/hip_runtime.h>
#include <iostream>
#include <kernel/kernel.hpp>
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

#define RUN_CONFIG(WM, WN, TM, TN)                                                                \
    do                                                                                            \
    {                                                                                             \
        int  block_m      = WM * TM * wmma_tile;                                                  \
        int  block_n      = WN * TN * wmma_tile;                                                  \
        int  grid_m       = (M + block_m - 1) / block_m;                                          \
        int  grid_n       = (N + block_n - 1) / block_n;                                          \
        int  total_blocks = grid_m * grid_n;                                                      \
        dim3 grid_dim(total_blocks, 1);                                                           \
        dim3 block_dim(warp_size* WM* WN);                                                        \
                                                                                                  \
        gpu_timer timer;                                                                          \
        float     total_time = 0.0f;                                                              \
                                                                                                  \
        /* Warmup */                                                                              \
        for(int i = 0; i < 10; ++i)                                                               \
        {                                                                                         \
            kernel_gemm<half, c_layout, a_layout, b_layout, WM, WN, TM, TN>                       \
                <<<grid_dim, block_dim, 0, stream>>>(d_C, d_A, d_B, M, N, K);                     \
            HIP_CHECK(hipPeekAtLastError());                                                      \
        }                                                                                         \
        HIP_CHECK(hipStreamSynchronize(stream));                                                  \
                                                                                                  \
        /* Timing */                                                                              \
        for(int i = 0; i < ITERATIONS; ++i)                                                       \
        {                                                                                         \
            timer.start(stream);                                                                  \
            kernel_gemm<half, c_layout, a_layout, b_layout, WM, WN, TM, TN>                       \
                <<<grid_dim, block_dim, 0, stream>>>(d_C, d_A, d_B, M, N, K);                     \
            HIP_CHECK(hipPeekAtLastError());                                                      \
            total_time += timer.stop(stream);                                                     \
            HIP_CHECK(hipDeviceSynchronize());                                                    \
        }                                                                                         \
        HIP_CHECK(hipDeviceSynchronize());                                                        \
                                                                                                  \
        std::string layout_a = (a_layout == m_layout::row_major) ? "row_major" : "col_major";     \
        std::string layout_b = (b_layout == m_layout::row_major) ? "row_major" : "col_major";     \
        std::string layout_c = (c_layout == m_layout::row_major) ? "row_major" : "col_major";     \
                                                                                                  \
        std::cout << std::fixed << std::setprecision(3) << M << "," << N << "," << K << "," << WM \
                  << "," << WN << "," << TM << "," << TN << "," << layout_a << "," << layout_b    \
                  << "," << layout_c << "," << (total_time / ITERATIONS) << "," << total_time     \
                  << std::endl;                                                                   \
    }                                                                                             \
    while(0)

template<m_layout a_layout, m_layout b_layout, m_layout c_layout>
void tune_size(size_t M, size_t N, size_t K)
{
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<half> h_C(M * N);

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

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    half *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(half)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), K * N * sizeof(half), hipMemcpyHostToDevice));

    constexpr int ITERATIONS = 20;

    // Test all combinations with compile-time configurations
    RUN_CONFIG(2, 2, 2, 2);
    RUN_CONFIG(2, 2, 2, 4);
    RUN_CONFIG(2, 2, 4, 2);
    RUN_CONFIG(2, 2, 4, 4);
    RUN_CONFIG(2, 4, 2, 2);
    RUN_CONFIG(2, 4, 2, 4);
    RUN_CONFIG(2, 4, 4, 2);
    RUN_CONFIG(2, 4, 4, 4);
    RUN_CONFIG(4, 2, 2, 2);
    RUN_CONFIG(4, 2, 2, 4);
    RUN_CONFIG(4, 2, 4, 2);
    RUN_CONFIG(4, 2, 4, 4);
    RUN_CONFIG(4, 4, 2, 2);
    RUN_CONFIG(4, 4, 2, 4);
    RUN_CONFIG(4, 4, 4, 2);
    RUN_CONFIG(4, 4, 4, 4);

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " M N K" << std::endl;
        return 1;
    }

    size_t M = std::atol(argv[1]);
    size_t N = std::atol(argv[2]);
    size_t K = std::atol(argv[3]);

    std::cout << "Testing size " << M << "x" << N << "x" << K << std::endl;

    tune_size<m_layout::row_major, m_layout::row_major, m_layout::row_major>(M, N, K);
    tune_size<m_layout::row_major, m_layout::col_major, m_layout::row_major>(M, N, K);
    tune_size<m_layout::col_major, m_layout::row_major, m_layout::row_major>(M, N, K);
    tune_size<m_layout::col_major, m_layout::col_major, m_layout::row_major>(M, N, K);
    tune_size<m_layout::row_major, m_layout::row_major, m_layout::col_major>(M, N, K);
    tune_size<m_layout::row_major, m_layout::col_major, m_layout::col_major>(M, N, K);
    tune_size<m_layout::col_major, m_layout::row_major, m_layout::col_major>(M, N, K);
    tune_size<m_layout::col_major, m_layout::col_major, m_layout::col_major>(M, N, K);

    return 0;
}
