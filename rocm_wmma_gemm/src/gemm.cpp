#include <hip/hip_runtime.h>
#include <rocm_wmma_gemm/gemm.hpp>
#include <rocm_wmma_gemm/kernel_launcher.hpp>

namespace rocm_wmma_gemm
{

template<m_layout layout_C, m_layout layout_A, m_layout layout_B>
__host__ void gemm(half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    gemm<layout_C, layout_A, layout_B>(C, A, B, M, N, K, 1, stream);
}

template<m_layout layout_C, m_layout layout_A, m_layout layout_B>
__host__ void gemm(half*        C,
                   half*        A,
                   half*        B,
                   size_t       M,
                   size_t       N,
                   size_t       K,
                   size_t       batch_count,
                   hipStream_t& stream)
{
    // Get optimal parameters from generated configuration
    auto params = get_gemm_params(M, N, K, layout_C, layout_A, layout_B);

    int block_m = params.warps_m * params.warp_tile_m * wmma_tile;
    int block_n = params.warps_n * params.warp_tile_n * wmma_tile;

    // Calculate grid dimensions
    int grid_m       = (M + block_m - 1) / block_m;
    int grid_n       = (N + block_n - 1) / block_n;
    int total_blocks = grid_m * grid_n;

    dim3 grid_dim(total_blocks, batch_count);
    dim3 block_dim(warp_size * params.warps_m * params.warps_n);

    // Launch kernel using the template launcher
    kernel_launcher<half, half, layout_C, layout_A, layout_B>::launch(params,
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

template __host__ void gemm<m_layout::row_major, m_layout::col_major, m_layout::col_major>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void gemm<m_layout::row_major, m_layout::row_major, m_layout::col_major>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void gemm<m_layout::row_major, m_layout::col_major, m_layout::row_major>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void gemm<m_layout::row_major, m_layout::row_major, m_layout::row_major>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void gemm<m_layout::col_major, m_layout::col_major, m_layout::col_major>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void gemm<m_layout::col_major, m_layout::row_major, m_layout::col_major>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void gemm<m_layout::col_major, m_layout::col_major, m_layout::row_major>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void gemm<m_layout::col_major, m_layout::row_major, m_layout::row_major>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void
    gemm<m_layout::row_major, m_layout::col_major, m_layout::col_major>(half*        C,
                                                                        half*        A,
                                                                        half*        B,
                                                                        size_t       M,
                                                                        size_t       N,
                                                                        size_t       K,
                                                                        size_t       batch_count,
                                                                        hipStream_t& stream);

template __host__ void
    gemm<m_layout::row_major, m_layout::row_major, m_layout::col_major>(half*        C,
                                                                        half*        A,
                                                                        half*        B,
                                                                        size_t       M,
                                                                        size_t       N,
                                                                        size_t       K,
                                                                        size_t       batch_count,
                                                                        hipStream_t& stream);

template __host__ void
    gemm<m_layout::row_major, m_layout::col_major, m_layout::row_major>(half*        C,
                                                                        half*        A,
                                                                        half*        B,
                                                                        size_t       M,
                                                                        size_t       N,
                                                                        size_t       K,
                                                                        size_t       batch_count,
                                                                        hipStream_t& stream);

template __host__ void
    gemm<m_layout::row_major, m_layout::row_major, m_layout::row_major>(half*        C,
                                                                        half*        A,
                                                                        half*        B,
                                                                        size_t       M,
                                                                        size_t       N,
                                                                        size_t       K,
                                                                        size_t       batch_count,
                                                                        hipStream_t& stream);

template __host__ void
    gemm<m_layout::col_major, m_layout::col_major, m_layout::col_major>(half*        C,
                                                                        half*        A,
                                                                        half*        B,
                                                                        size_t       M,
                                                                        size_t       N,
                                                                        size_t       K,
                                                                        size_t       batch_count,
                                                                        hipStream_t& stream);

template __host__ void
    gemm<m_layout::col_major, m_layout::row_major, m_layout::col_major>(half*        C,
                                                                        half*        A,
                                                                        half*        B,
                                                                        size_t       M,
                                                                        size_t       N,
                                                                        size_t       K,
                                                                        size_t       batch_count,
                                                                        hipStream_t& stream);

template __host__ void
    gemm<m_layout::col_major, m_layout::col_major, m_layout::row_major>(half*        C,
                                                                        half*        A,
                                                                        half*        B,
                                                                        size_t       M,
                                                                        size_t       N,
                                                                        size_t       K,
                                                                        size_t       batch_count,
                                                                        hipStream_t& stream);

template __host__ void
    gemm<m_layout::col_major, m_layout::row_major, m_layout::row_major>(half*        C,
                                                                        half*        A,
                                                                        half*        B,
                                                                        size_t       M,
                                                                        size_t       N,
                                                                        size_t       K,
                                                                        size_t       batch_count,
                                                                        hipStream_t& stream);

} // namespace rocm_wmma_gemm
