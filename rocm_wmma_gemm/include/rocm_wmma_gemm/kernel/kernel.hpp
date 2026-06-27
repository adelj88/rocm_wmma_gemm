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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ROCM_WMMA_KERNEL_HPP
#define ROCM_WMMA_KERNEL_HPP

#include "common.hpp"
#include "fragment.hpp"
#include "load.hpp"
#include "mapping.hpp"
#include "wmma.hpp"

namespace rocm_wmma_gemm
{

/**
 * @brief Base configuration struct for wave (warp) settings.
 */
template<int warps_m, int warps_n>
struct wave_config_base
{
    static constexpr int total_warps = warps_m * warps_n; ///< Total number of warps per block.
};

/**
 * @brief Configuration struct for tuning AMD wave bounds `__launch_bounds__`.
 *
 * Provides minimum and maximum waves per execution unit to optimize register usage
 * and occupancy for the given matrix layouts.
 */
template<m_layout LAYOUT_A, m_layout LAYOUT_B, int warps_m, int warps_n>
struct wave_config : wave_config_base<warps_m, warps_n>
{
    using base               = wave_config_base<warps_m, warps_n>;
    static constexpr int min = 2;
    static constexpr int max = 8;
};

/**
 * @brief Wave configuration specialization for row-major A and col-major B.
 */
template<int warps_m, int warps_n>
struct wave_config<m_layout::row_major, m_layout::col_major, warps_m, warps_n>
    : wave_config_base<warps_m, warps_n>
{
    using base               = wave_config_base<warps_m, warps_n>;
    static constexpr int min = 2;
    static constexpr int max = 4;
};

/**
 * @brief Core device function implementing the WMMA-based GEMM operation.
 *
 * This function computes a block of the output matrix C. It employs several advanced
 * optimizations for performance:
 *
 * 1. **Software Pipelining (Double Buffering)**: It uses two buffers in shared memory (LDS)
 *    to overlap global memory loads with compute. While warps compute the outer product
 *    of the current tiles of A and B from LDS, memory instructions fetch the next tiles
 *    from global memory into registers, and then commit them to the alternate LDS buffer.
 *
 * 2. **Register Prefetching**: Data is staged in registers via `prefetch_fragment` before
 *    being committed to LDS. This creates a multi-stage pipeline (Global -> Registers -> LDS)
 *    that hides memory latency. `partial_prefetch` interleaves these loads with WMMA math.
 *
 * 3. **Warp Tiling**: Each thread block is divided into warps (`warps_m` x `warps_n`).
 *    Each warp computes a specific sub-tile of the block (`warp_tile_m` x `warp_tile_n`
 *    WMMA fragments). This hierarchy maximizes data reuse in LDS and registers.
 *
 * 4. **Chunked Epilogue**: After the main K-loop finishes, the accumulated C fragments
 *    residing in warp registers must be written to global memory. To ensure coalesced writes
 *    and avoid LDS bank conflicts, the fragments are first stored back to the LDS buffers
 *    in chunks, and then the entire thread block cooperatively writes those chunks to global memory.
 *
 * @tparam T Data type of output matrix C.
 * @tparam U Data type of input matrices A and B.
 * @tparam LAYOUT_C Memory layout of C.
 * @tparam LAYOUT_A Memory layout of A.
 * @tparam LAYOUT_B Memory layout of B.
 * @tparam warps_m Number of warps mapped to the M dimension.
 * @tparam warps_n Number of warps mapped to the N dimension.
 * @tparam warp_tile_m Number of WMMA tiles per warp in the M dimension.
 * @tparam warp_tile_n Number of WMMA tiles per warp in the N dimension.
 * @tparam swizzle Swizzle size for the block mapping.
 * @tparam bits Vectorization bit width for memory loads.
 */
template<class T,
         class U,
         m_layout LAYOUT_C,
         m_layout LAYOUT_A,
         m_layout LAYOUT_B,
         int      warps_m,
         int      warps_n,
         int      warp_tile_m,
         int      warp_tile_n,
         int      swizzle,
         int      bits,
         int      is_aligned>
__device__ __forceinline__ void gemm_impl(
    T* __restrict__ C, const U* __restrict__ A, const U* __restrict__ B, int M, int N, int K)
{
    static_assert(warps_m != 0 && warps_n != 0);
    static_assert((warp_tile_m * warp_tile_n) > 1);

    // Conditional padding
    constexpr int padding_a = (LAYOUT_A == m_layout::row_major) ? 8 : 0;
    constexpr int padding_b = (LAYOUT_B == m_layout::col_major) ? 8 : 0;

    constexpr int block_m = warps_m * warp_tile_m * wmma_tile;
    constexpr int block_n = warps_n * warp_tile_n * wmma_tile;
    constexpr int block_k = wmma_tile;

    constexpr int stride_a = block_k + padding_a;
    constexpr int stride_b = block_k + padding_b;

    // Shared memory size with padding
    constexpr int lds_size = (block_m * stride_a) + (stride_b * block_n);

    // Calculate grid dimensions
    const int grid_m  = (M + block_m - 1) / block_m;
    const int grid_n  = (N + block_n - 1) / block_n;
    const int tile_id = blockIdx.x;

    using mapper = tile_mapper<block_m, block_n, LAYOUT_A, LAYOUT_B, swizzle>;

    // Get block coordinates
    int block_row, block_col;
    mapper().map_tile(tile_id, grid_m, grid_n, &block_row, &block_col);

    // 2x LDS always — used for both compute/next buffering and output epilogue
    __shared__ U lds_mem[2 * lds_size];

    U* a_tiles_0 = lds_mem;
    U* a_tiles_1 = lds_mem + lds_size;
    U* b_tiles_0 = lds_mem + (block_m * stride_a);
    U* b_tiles_1 = lds_mem + lds_size + (block_m * stride_a);

    // Each block is launched with a one-dimensional thread block.
    constexpr int full_block = warp_size * warps_m * warps_n;
    constexpr int half_block = full_block / 2;
    const int     tid        = threadIdx.x;
    const int     cid        = tid % half_block;

    A += blockIdx.y * M * K;
    B += blockIdx.y * K * N;
    C += blockIdx.y * M * N;

    const U* A_base = A + block_row * ((LAYOUT_A == m_layout::col_major) ? 1 : K);
    const U* B_base = B + block_col * ((LAYOUT_B == m_layout::col_major) ? K : 1);

    // Compute warp ID from the 1D thread index.
    const int warp_id  = tid / warp_size;
    const int warp_row = warp_id / warps_n;
    const int warp_col = warp_id % warps_n;

    constexpr int half_warp    = warp_size / 2;
    const int     lane_id      = (tid % warp_size);
    const int     half_warp_id = lane_id / half_warp;
    const int     half_lane    = tid % half_warp;

    // Determine the base offsets for this warp's set of WMMA tiles.
    const int warp_m_base = warp_row * warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * warp_tile_n * wmma_tile;

    // Declare fragment storage.
    fragment<T, wmma_tile> c_frags[warp_tile_m][warp_tile_n];
    fragment<U, wmma_tile> a_frag[warp_tile_m];
    fragment<U, wmma_tile> b_frag[warp_tile_n];

    // Prefetch fragments — registers stage global loads before commit to LDS
    prefetch_fragment<LAYOUT_A, bits, full_block, block_m, block_k, padding_a, U> regs_a;
    prefetch_fragment<LAYOUT_B, bits, full_block, block_k, block_n, padding_b, U> regs_b;

    // Base pointers for the current A and B tiles.
    const U* A_tile_ptr = A_base;
    const U* B_tile_ptr = B_base;

    const int global_mult_A = block_k * ((LAYOUT_A == m_layout::col_major) ? M : 1);
    const int global_mult_B = block_k * ((LAYOUT_B == m_layout::col_major) ? 1 : N);

    // Fragment multipliers
    constexpr int frag_mult_A = (LAYOUT_A == m_layout::col_major) ? 1 : stride_a;
    constexpr int frag_mult_B = (LAYOUT_B == m_layout::col_major) ? stride_b : 1;

    constexpr int frag_offset_A = wmma_tile * frag_mult_A;
    constexpr int frag_offset_B = wmma_tile * frag_mult_B;

    const int warp_offset_A = (warp_m_base + half_lane) * frag_mult_A;
    const int warp_offset_B = (warp_n_base + half_lane) * frag_mult_B;

    // Initial load into current LDS buffer
    regs_a.prefetch(A_tile_ptr, M, K, tid);
    regs_b.prefetch(B_tile_ptr, K, N, tid);
    regs_a.commit(a_tiles_0, tid);
    regs_b.commit(b_tiles_0, tid);
    __syncthreads();

    U* current_a = a_tiles_0;
    U* current_b = b_tiles_0;
    U* next_a    = a_tiles_1;
    U* next_b    = b_tiles_1;

    constexpr bool   warp_m_is_major = warp_tile_m >= warp_tile_n;
    constexpr size_t warp_outer_max  = warp_m_is_major ? warp_tile_m : warp_tile_n;
    constexpr size_t warp_inner_max  = warp_m_is_major ? warp_tile_n : warp_tile_m;
    constexpr size_t num_combos      = warp_outer_max * warp_inner_max;
    constexpr size_t num_slices      = num_combos / 2;

    constexpr auto get_wm = [](size_t i) constexpr -> size_t
    {
        size_t w_o   = i / warp_inner_max;
        size_t w_i   = i % warp_inner_max;
        size_t w_i_s = (w_o & 1) ? (warp_inner_max - w_i - 1) : w_i;
        return warp_m_is_major ? w_o : w_i_s;
    };

    constexpr auto get_wn = [](size_t i) constexpr -> size_t
    {
        size_t w_o   = i / warp_inner_max;
        size_t w_i   = i % warp_inner_max;
        size_t w_i_s = (w_o & 1) ? (warp_inner_max - w_i - 1) : w_i;
        return warp_m_is_major ? w_i_s : w_o;
    };

    // Main loop over k-dimension
    for(int k_tile = 0; k_tile < K - block_k; k_tile += block_k)
    {
        const U* curr_a = current_a + warp_offset_A;
        const U* curr_b = current_b + warp_offset_B;

        const U* next_A = A_tile_ptr + global_mult_A;
        const U* next_B = B_tile_ptr + global_mult_B;

        auto load_a_frag = [&]<size_t idx>()
        {
            load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[idx],
                                                     curr_a + idx * frag_offset_A,
                                                     block_m,
                                                     stride_a);
        };

        auto load_b_frag = [&]<size_t idx>()
        {
            load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[idx],
                                                     curr_b + idx * frag_offset_B,
                                                     stride_b,
                                                     block_n);
        };

        if constexpr(warp_m_is_major)
        {
            load_a_frag.template operator()<0>();
            load_b_frag.template operator()<0>();
        }
        else
        {
            load_b_frag.template operator()<0>();
            load_a_frag.template operator()<0>();
        }

        [&]<size_t... i>(std::index_sequence<i...>)
        {
            (
                [&]()
                {
                    constexpr size_t wm = get_wm(i);
                    constexpr size_t wn = get_wn(i);

                    constexpr size_t next_i = i + 1;
                    if constexpr(next_i < num_combos)
                    {
                        constexpr size_t next_wm = get_wm(next_i);
                        constexpr size_t next_wn = get_wn(next_i);

                        if constexpr(warp_m_is_major)
                        {
                            if constexpr(next_wm != wm)
                            {
                                load_a_frag.template operator()<next_wm>();
                            }
                            if constexpr(next_wn != wn && next_wm == 0)
                            {
                                load_b_frag.template operator()<next_wn>();
                            }
                        }
                        else
                        {
                            if constexpr(next_wn != wn)
                            {
                                load_b_frag.template operator()<next_wn>();
                            }
                            if constexpr(next_wm != wm && next_wn == 0)
                            {
                                load_a_frag.template operator()<next_wm>();
                            }
                        }
                    }

                    constexpr size_t slice = i / 2;
                    if constexpr((i % 2) == 0)
                    {
                        // Even iterations: prefetch A
                        regs_a.template partial_prefetch<slice, num_slices>(next_A, M, K, tid);
                    }
                    else
                    {
                        // Odd iterations: prefetch B
                        regs_b.template partial_prefetch<slice, num_slices>(next_B, K, N, tid);
                    }

                    wmma(a_frag[wm], b_frag[wn], c_frags[wm][wn]);
                }(),
                ...);
        }(std::make_index_sequence<num_combos>{});

        regs_a.commit(next_a, tid);
        regs_b.commit(next_b, tid);

        // Swap buffer pointers
        U* temp_a = current_a;
        U* temp_b = current_b;
        current_a = next_a;
        current_b = next_b;
        next_a    = temp_a;
        next_b    = temp_b;

        A_tile_ptr += global_mult_A;
        B_tile_ptr += global_mult_B;
        __syncthreads();
    }

    // Epilogue for the final K tile
    const U* curr_a = current_a + warp_offset_A;
    const U* curr_b = current_b + warp_offset_B;

    auto load_a_frag = [&]<size_t idx>()
    {
        load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[idx],
                                                 curr_a + idx * frag_offset_A,
                                                 block_m,
                                                 stride_a);
    };

    auto load_b_frag = [&]<size_t idx>()
    {
        load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[idx],
                                                 curr_b + idx * frag_offset_B,
                                                 stride_b,
                                                 block_n);
    };

    if constexpr(warp_m_is_major)
    {
        load_a_frag.template operator()<0>();
        load_b_frag.template operator()<0>();
    }
    else
    {
        load_b_frag.template operator()<0>();
        load_a_frag.template operator()<0>();
    }

    [&]<size_t... i>(std::index_sequence<i...>)
    {
        (
            [&]()
            {
                constexpr size_t wm = get_wm(i);
                constexpr size_t wn = get_wn(i);

                constexpr size_t next_i = i + 1;
                if constexpr(next_i < num_combos)
                {
                    constexpr size_t next_wm = get_wm(next_i);
                    constexpr size_t next_wn = get_wn(next_i);

                    if constexpr(warp_m_is_major)
                    {
                        if constexpr(next_wm != wm)
                        {
                            load_a_frag.template operator()<next_wm>();
                        }
                        if constexpr(next_wn != wn && next_wm == 0)
                        {
                            load_b_frag.template operator()<next_wn>();
                        }
                    }
                    else
                    {
                        if constexpr(next_wn != wn)
                        {
                            load_b_frag.template operator()<next_wn>();
                        }
                        if constexpr(next_wm != wm && next_wn == 0)
                        {
                            load_a_frag.template operator()<next_wm>();
                        }
                    }
                }

                wmma(a_frag[wm], b_frag[wn], c_frags[wm][wn]);
            }(),
            ...);
    }(std::make_index_sequence<num_combos>{});

    __syncthreads();

    constexpr bool is_col_major = (LAYOUT_C == m_layout::col_major);

    constexpr int chunk_rows = wmma_tile;
    constexpr int chunk_cols = wmma_tile;

    constexpr int num_chunks = is_col_major ? (warps_n * warp_tile_n) : (warps_m * warp_tile_m);

    constexpr int available_bytes   = 2 * lds_size * static_cast<int>(sizeof(U));
    constexpr int chunk_slice_bytes = is_col_major
                                          ? (block_m * chunk_cols * static_cast<int>(sizeof(T)))
                                          : (chunk_rows * block_n * static_cast<int>(sizeof(T)));

    // Store two chunks per LDS buffer when they fit, halving sync and flush count.
    constexpr bool can_pair = (2 * chunk_slice_bytes <= available_bytes);

    T* c_buf = reinterpret_cast<T*>(lds_mem);

    auto store_chunk = [&]<size_t ci>(T* buf, int local_offset)
    {
        if constexpr(is_col_major)
        {
            constexpr int responsible_warp_col = static_cast<int>(ci) / warp_tile_n;
            constexpr int wn_idx               = static_cast<int>(ci) % warp_tile_n;
            constexpr int buf_cols             = can_pair ? 2 * chunk_cols : chunk_cols;

            if(warp_col == responsible_warp_col)
            {
                [&]<size_t... wm>(std::index_sequence<wm...>)
                {
                    (store_matrix<LAYOUT_C, true>(buf,
                                                  c_frags[wm][wn_idx],
                                                  warp_m_base + static_cast<int>(wm) * wmma_tile
                                                      + half_warp_id,
                                                  local_offset + half_lane,
                                                  block_m,
                                                  buf_cols),
                     ...);
                }(std::make_index_sequence<warp_tile_m>{});
            }
        }
        else
        {
            constexpr int responsible_warp_row = static_cast<int>(ci) / warp_tile_m;
            constexpr int wm_idx               = static_cast<int>(ci) % warp_tile_m;
            constexpr int buf_rows             = can_pair ? 2 * chunk_rows : chunk_rows;

            if(warp_row == responsible_warp_row)
            {
                [&]<size_t... wn>(std::index_sequence<wn...>)
                {
                    (store_matrix<LAYOUT_C, true>(buf,
                                                  c_frags[wm_idx][wn],
                                                  local_offset + half_warp_id,
                                                  warp_n_base + static_cast<int>(wn) * wmma_tile
                                                      + half_lane,
                                                  buf_rows,
                                                  block_n),
                     ...);
                }(std::make_index_sequence<warp_tile_n>{});
            }
        }
    };

    auto flush_chunk = [&]<size_t ci, int STEP>(T* buf)
    {
        if constexpr(is_col_major)
        {
            load_shared_to_global<LAYOUT_C, bits, full_block, block_m, STEP, is_aligned>(
                C,
                buf,
                block_row,
                block_col + static_cast<int>(ci) * wmma_tile,
                M,
                N,
                tid);
        }
        else
        {
            load_shared_to_global<LAYOUT_C, bits, full_block, STEP, block_n, is_aligned>(
                C,
                buf,
                block_row + static_cast<int>(ci) * wmma_tile,
                block_col,
                M,
                N,
                tid);
        }
    };

    if constexpr(can_pair)
    {
        static_assert(num_chunks % 2 == 0,
                      "num_chunks must be even (warps and warp_tiles are powers of two)");
        constexpr int num_pairs = num_chunks / 2;

        [&]<size_t... pi>(std::index_sequence<pi...>)
        {
            (
                [&]()
                {
                    constexpr size_t ci0 = pi * 2;
                    constexpr size_t ci1 = ci0 + 1;

                    store_chunk.template operator()<ci0>(c_buf, 0);
                    store_chunk.template operator()<ci1>(c_buf, wmma_tile);
                    __syncthreads();
                    flush_chunk.template operator()<ci0, 2 * wmma_tile>(c_buf);
                    __syncthreads();
                }(),
                ...);
        }(std::make_index_sequence<num_pairs>{});
    }
    else
    {
        [&]<size_t... ci>(std::index_sequence<ci...>)
        {
            (
                [&]()
                {
                    store_chunk.template operator()<ci>(c_buf, 0);
                    __syncthreads();
                    flush_chunk.template operator()<ci, wmma_tile>(c_buf);
                    __syncthreads();
                }(),
                ...);
        }(std::make_index_sequence<num_chunks>{});
    }
}

/**
 * @brief Functor wrapping the GEMM kernel launch.
 */
template<class T,
         class U,
         m_layout LAYOUT_C,
         m_layout LAYOUT_A,
         m_layout LAYOUT_B,
         int      warps_m,
         int      warps_n,
         int      warp_tile_m,
         int      warp_tile_n,
         int      swizzle,
         int      bits,
         int      is_aligned>
struct kernel_gemm_impl
{
    using config = wave_config<LAYOUT_A, LAYOUT_B, warps_m, warps_n>;

    /**
     * @brief The global entry point for the GEMM kernel.
     */
    __global__ __launch_bounds__(warp_size* warps_m* warps_n)
        __attribute__((amdgpu_waves_per_eu(config::min,
                                           config::max))) static void run(T* __restrict__ C,
                                                                          const U* __restrict__ A,
                                                                          const U* __restrict__ B,
                                                                          int M,
                                                                          int N,
                                                                          int K)
    {
        gemm_impl<T,
                  U,
                  LAYOUT_C,
                  LAYOUT_A,
                  LAYOUT_B,
                  warps_m,
                  warps_n,
                  warp_tile_m,
                  warp_tile_n,
                  swizzle,
                  bits,
                  is_aligned>(C, A, B, M, N, K);
    }
};

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_KERNEL_HPP
