#include <hip/hip_runtime.h>
#include <rocm_wmma_gemm/gemm.hpp>
#include <rocm_wmma_gemm/kernel_launcher.hpp>

namespace rocm_wmma_gemm
{

template<m_layout layout_C, m_layout layout_A, m_layout layout_B, class T, class U>
__host__ void gemm(T* C, U* A, U* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    gemm<layout_C, layout_A, layout_B>(C, A, B, M, N, K, 1, stream);
}

template<m_layout layout_C, m_layout layout_A, m_layout layout_B, class T, class U>
__host__ void
    gemm(T* C, U* A, U* B, size_t M, size_t N, size_t K, size_t batch_count, hipStream_t& stream)
{
    // Find the best config index for this problem size and layout
    size_t config_idx = detail::find_best_config(M, N, K, layout_A, layout_B, layout_C);

    // Get the params for grid/block calculation
    const auto& config      = detail::kernel_configs[config_idx];
    int         warps_m     = std::get<0>(config);
    int         warps_n     = std::get<1>(config);
    int         warp_tile_m = std::get<2>(config);
    int         warp_tile_n = std::get<3>(config);

    int block_m = warps_m * warp_tile_m * wmma_tile;
    int block_n = warps_n * warp_tile_n * wmma_tile;

    int grid_m       = (M + block_m - 1) / block_m;
    int grid_n       = (N + block_n - 1) / block_n;
    int total_blocks = grid_m * grid_n;

    dim3 grid_dim(total_blocks, batch_count);
    dim3 block_dim(warp_size * warps_m * warps_n);

    kernel_launcher<T, U, layout_C, layout_A, layout_B>::launch(config_idx,
                                                                C,
                                                                A,
                                                                B,
                                                                M,
                                                                N,
                                                                K,
                                                                grid_dim,
                                                                block_dim,
                                                                stream);
}

// Macro to instantiate all layout combinations for a type pair
#define INSTANTIATE_GEMM_FOR_TYPES(T, U)                                               \
    template void gemm<m_layout::row_major, m_layout::row_major, m_layout::row_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::row_major, m_layout::row_major, m_layout::col_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::row_major, m_layout::col_major, m_layout::row_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::row_major, m_layout::col_major, m_layout::col_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::col_major, m_layout::row_major, m_layout::row_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::col_major, m_layout::row_major, m_layout::col_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::col_major, m_layout::col_major, m_layout::row_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::col_major, m_layout::col_major, m_layout::col_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::row_major, m_layout::row_major, m_layout::row_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::row_major, m_layout::row_major, m_layout::col_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::row_major, m_layout::col_major, m_layout::row_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::row_major, m_layout::col_major, m_layout::col_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::col_major, m_layout::row_major, m_layout::row_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::col_major, m_layout::row_major, m_layout::col_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::col_major, m_layout::col_major, m_layout::row_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::col_major, m_layout::col_major, m_layout::col_major>( \
        T*,                                                                            \
        U*,                                                                            \
        U*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);

// Instantiate the types you want
INSTANTIATE_GEMM_FOR_TYPES(half, half)
INSTANTIATE_GEMM_FOR_TYPES(float, half)
INSTANTIATE_GEMM_FOR_TYPES(__hip_bfloat16, __hip_bfloat16)
INSTANTIATE_GEMM_FOR_TYPES(float, __hip_bfloat16)

} // namespace rocm_wmma_gemm
