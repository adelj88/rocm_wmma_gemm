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
    // type_group: 0 = f16/bf16 accumulator, 1 = f32 accumulator
    constexpr size_t type_group = std::is_same_v<T, float> ? 1 : 0;

    // Find the best config index for this problem size, type group, and layout
    size_t config_idx = detail::find_best_config(M, N, K, type_group, layout_A, layout_B, layout_C);

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
                                                                block_m,
                                                                block_n,
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

// =============================================================================
// C ABI entry points for runtime arch-library loading (used by kernel_loader).
// layout_a/b/c: 0 = row_major, 1 = col_major.
// =============================================================================

#define DISPATCH_LAYOUT(T, U, lc, la, lb)                                  \
    rocm_wmma_gemm::gemm<rocm_wmma_gemm::m_layout::lc,                     \
                         rocm_wmma_gemm::m_layout::la,                     \
                         rocm_wmma_gemm::m_layout::lb>(static_cast<T*>(C), \
                                                       static_cast<U*>(A), \
                                                       static_cast<U*>(B), \
                                                       M,                  \
                                                       N,                  \
                                                       K,                  \
                                                       batch_count,        \
                                                       *static_cast<hipStream_t*>(stream))

#define DEFINE_C_DISPATCH(suffix, T, U)                                            \
    extern "C" void rocm_wmma_gemm_##suffix(void*  C,                              \
                                            void*  A,                              \
                                            void*  B,                              \
                                            size_t M,                              \
                                            size_t N,                              \
                                            size_t K,                              \
                                            size_t batch_count,                    \
                                            int    layout_c,                       \
                                            int    layout_a,                       \
                                            int    layout_b,                       \
                                            void*  stream)                         \
    {                                                                              \
        const int key = layout_c << 2 | layout_a << 1 | layout_b;                  \
        switch(key)                                                                \
        {                                                                          \
            case 0: DISPATCH_LAYOUT(T, U, row_major, row_major, row_major); break; \
            case 1: DISPATCH_LAYOUT(T, U, row_major, row_major, col_major); break; \
            case 2: DISPATCH_LAYOUT(T, U, row_major, col_major, row_major); break; \
            case 3: DISPATCH_LAYOUT(T, U, row_major, col_major, col_major); break; \
            case 4: DISPATCH_LAYOUT(T, U, col_major, row_major, row_major); break; \
            case 5: DISPATCH_LAYOUT(T, U, col_major, row_major, col_major); break; \
            case 6: DISPATCH_LAYOUT(T, U, col_major, col_major, row_major); break; \
            case 7: DISPATCH_LAYOUT(T, U, col_major, col_major, col_major); break; \
            default: break;                                                        \
        }                                                                          \
    }

DEFINE_C_DISPATCH(f16_f16, half, half)
DEFINE_C_DISPATCH(f32_f16, float, half)
DEFINE_C_DISPATCH(bf16_bf16, __hip_bfloat16, __hip_bfloat16)
DEFINE_C_DISPATCH(f32_bf16, float, __hip_bfloat16)
