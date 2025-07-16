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
    auto params = get_gemm_params(M, N, K, layout_C, layout_A, layout_B);

    int block_m = params.warps_m * params.warp_tile_m * wmma_tile;
    int block_n = params.warps_n * params.warp_tile_n * wmma_tile;

    int grid_m       = (M + block_m - 1) / block_m;
    int grid_n       = (N + block_n - 1) / block_n;
    int total_blocks = grid_m * grid_n;

    dim3 grid_dim(total_blocks, batch_count);
    dim3 block_dim(warp_size * params.warps_m * params.warps_n);

    kernel_launcher<T, U, layout_C, layout_A, layout_B>::launch(params,
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
