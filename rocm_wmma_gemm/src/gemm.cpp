#include <gemm.hpp>
#include <hip/hip_runtime.h>
#include <kernel/kernel.hpp>

namespace rocm_wmma_gemm
{

template<m_layout layout_C, m_layout layout_A, m_layout layout_B>
__host__ void gemm(half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    int warps_m     = 4;
    int warps_n     = 4;
    int warp_tile_m = 4;
    int warp_tile_n = 4;

    if(!(layout_A == m_layout::row_major && layout_B == m_layout::col_major) && M <= 1024
       && N <= 1024)
    {
        warps_m     = 4;
        warps_n     = 2;
        warp_tile_m = 2;
        warp_tile_n = 2;
    }
    else if((layout_A == m_layout::row_major && layout_B == m_layout::col_major) && M <= 1024
            && N <= 1024)
    {
        warps_m     = 4;
        warps_n     = 4;
        warp_tile_m = 2;
        warp_tile_n = 2;
    }
    else if(!((layout_A == m_layout::col_major && layout_B == m_layout::row_major)
              || (layout_A == m_layout::row_major && layout_B == m_layout::row_major))
            && M > 1024 && M <= 2048 && N > 1024 && N <= 2048)
    {
        warps_m     = 4;
        warps_n     = 4;
        warp_tile_m = 4;
        warp_tile_n = 2;
    }

    int total_warps = warps_m * warps_n;
    int block_m     = warps_m * warp_tile_m * wmma_tile; // 4*4*16 = 256
    int block_n     = warps_n * warp_tile_n * wmma_tile; // 4*4*16 = 256

    // Calculate grid dimensions
    int grid_m       = (M + block_m - 1) / block_m;
    int grid_n       = (N + block_n - 1) / block_n;
    int total_blocks = grid_m * grid_n;

    dim3 grid_dim(total_blocks);
    dim3 block_dim(warp_size * total_warps);

    if(!(layout_A == m_layout::row_major && layout_B == m_layout::col_major) && M <= 1024
       && N <= 1024)
    {
        kernel_gemm<half, layout_C, layout_A, layout_B, 4, 2, 2, 2>
            <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
    }
    else if((layout_A == m_layout::row_major && layout_B == m_layout::col_major) && M <= 1024
            && N <= 1024)
    {
        kernel_gemm<half, layout_C, layout_A, layout_B, 4, 4, 2, 2>
            <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
    }
    else if(!((layout_A == m_layout::col_major && layout_B == m_layout::row_major)
              || (layout_A == m_layout::row_major && layout_B == m_layout::row_major))
            && M > 1024 && M <= 2048 && N > 1024 && N <= 2048)
    {
        kernel_gemm<half, layout_C, layout_A, layout_B, 4, 4, 4, 2>
            <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
    }
    else
    {
        kernel_gemm<half, layout_C, layout_A, layout_B, 4, 4, 4, 4>
            <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
    }
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

} // namespace rocm_wmma_gemm
