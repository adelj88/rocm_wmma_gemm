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

template<int warps_m, int warps_n>
struct wave_config_base
{
    static constexpr int total_warps = warps_m * warps_n;
};

// Default configuration
template<m_layout LAYOUT_A, m_layout LAYOUT_B, int warps_m, int warps_n>
struct wave_config : wave_config_base<warps_m, warps_n>
{
    using base               = wave_config_base<warps_m, warps_n>;
    static constexpr int min = 4;
    static constexpr int max = 8;
};

// Row,col specialization
template<int warps_m, int warps_n>
struct wave_config<m_layout::row_major, m_layout::col_major, warps_m, warps_n>
    : wave_config_base<warps_m, warps_n>
{
    using base               = wave_config_base<warps_m, warps_n>;
    static constexpr int min = 2;
    static constexpr int max = 4;
};

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
         bool     buffer_first,
         bool     use_async,
         bool     use_direct_write>
__device__ __forceinline__ void gemm_impl(
    T* __restrict__ C, const U* __restrict__ A, const U* __restrict__ B, int M, int N, int K)
{
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
    constexpr int prefetch_block = use_async ? half_block : full_block;
    prefetch_fragment<LAYOUT_A, bits, prefetch_block, block_m, block_k, padding_a, U> regs_a;
    prefetch_fragment<LAYOUT_B, bits, prefetch_block, block_k, block_n, padding_b, U> regs_b;

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
    if constexpr(use_async)
    {
        if(tid < half_block)
        {
            regs_a.prefetch(A_tile_ptr, M, K, cid);
            regs_a.commit(a_tiles_0, cid);
        }
        else
        {
            regs_b.prefetch(B_tile_ptr, K, N, cid);
            regs_b.commit(b_tiles_0, cid);
        }
    }
    else
    {
        regs_a.prefetch(A_tile_ptr, M, K, tid);
        regs_b.prefetch(B_tile_ptr, K, N, tid);
        regs_a.commit(a_tiles_0, tid);
        regs_b.commit(b_tiles_0, tid);
    }
    __syncthreads();

    U* current_a = a_tiles_0;
    U* current_b = b_tiles_0;
    U* next_a    = a_tiles_1;
    U* next_b    = b_tiles_1;

    // Main loop over k-dimension
    for(int k_tile = 0; k_tile < K; k_tile += block_k)
    {
        const bool has_next = (k_tile + block_k < K);
        const U*   curr_a   = current_a + warp_offset_A;
        const U*   curr_b   = current_b + warp_offset_B;

        if(has_next)
        {
            if constexpr(use_async)
            {
                if(tid < half_block)
                {
                    const U* next_A = A_tile_ptr + global_mult_A;
                    regs_a.prefetch(next_A, M, K, cid);
                    if constexpr(buffer_first)
                    {
                        regs_a.commit(next_a, cid);
                    }
                }
                else
                {
                    const U* next_B = B_tile_ptr + global_mult_B;
                    regs_b.prefetch(next_B, K, N, cid);
                    if constexpr(buffer_first)
                    {
                        regs_b.commit(next_b, cid);
                    }
                }
            }
            else
            {
                const U* next_A = A_tile_ptr + global_mult_A;
                const U* next_B = B_tile_ptr + global_mult_B;
                regs_a.prefetch(next_A, M, K, tid);
                regs_b.prefetch(next_B, K, N, tid);

                if constexpr(buffer_first)
                {
                    regs_a.commit(next_a, tid);
                    regs_b.commit(next_b, tid);
                }
            }
        }

        constexpr size_t wt_max
            = static_cast<size_t>(warp_tile_m > warp_tile_n ? warp_tile_m : warp_tile_n);

        auto load_ab = [&]<size_t wt>()
        {
            // LDS fragment loads for current tile
            if constexpr(warp_tile_m >= warp_tile_n)
            {
                if constexpr(wt < warp_tile_m)
                {
                    load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[wt], curr_a, block_m, stride_a);
                    curr_a += frag_offset_A;
                }

                if constexpr(wt < warp_tile_n)
                {
                    load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[wt], curr_b, stride_b, block_n);
                    curr_b += frag_offset_B;
                }
            }
            else
            {
                if constexpr(wt < warp_tile_n)
                {
                    load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[wt], curr_b, stride_b, block_n);
                    curr_b += frag_offset_B;
                }

                if constexpr(wt < warp_tile_m)
                {
                    load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[wt], curr_a, block_m, stride_a);
                    curr_a += frag_offset_A;
                }
            }
        };

        [&]<size_t... wt>(std::index_sequence<wt...>)
        { (load_ab.template operator()<wt>(), ...); }(std::make_index_sequence<wt_max>{});

        // Compute
        if constexpr(warp_tile_m >= warp_tile_n)
        {
            auto compute_wm = [&]<size_t wm>()
            {
                auto compute_wn = [&]<size_t wn>()
                {
                    size_t wn_s = (wm & 1) ? (warp_tile_n - wn - 1) : wn;
                    wmma(a_frag[wm], b_frag[wn_s], c_frags[wm][wn_s]);
                };

                [&]<size_t... wn>(std::index_sequence<wn...>) {
                    (compute_wn.template operator()<wn>(), ...);
                }(std::make_index_sequence<warp_tile_n>{});
            };

            [&]<size_t... wm>(std::index_sequence<wm...>) {
                (compute_wm.template operator()<wm>(), ...);
            }(std::make_index_sequence<warp_tile_m>{});
        }
        else
        {
            auto compute_wn = [&]<size_t wn>()
            {
                auto compute_wm = [&]<size_t wm>()
                {
                    size_t wm_s = (wn & 1) ? (warp_tile_m - wm - 1) : wm;
                    wmma(a_frag[wm_s], b_frag[wn], c_frags[wm_s][wn]);
                };

                [&]<size_t... wm>(std::index_sequence<wm...>) {
                    (compute_wm.template operator()<wm>(), ...);
                }(std::make_index_sequence<warp_tile_m>{});
            };

            [&]<size_t... wn>(std::index_sequence<wn...>) {
                (compute_wn.template operator()<wn>(), ...);
            }(std::make_index_sequence<warp_tile_n>{});
        }

        if constexpr(buffer_first)
        {
            // Swap buffer pointers
            U* temp_a = current_a;
            U* temp_b = current_b;
            current_a = next_a;
            current_b = next_b;
            next_a    = temp_a;
            next_b    = temp_b;
        }
        else
        {
            // Commit registers into current LDS (single buffer path)
            // syncthreads before commit guards load_matrix reads still in flight on other threads
            asm volatile("s_waitcnt vmcnt(0)");
            __builtin_amdgcn_s_barrier();
            if(has_next)
            {
                if constexpr(use_async)
                {
                    if(tid < half_block)
                    {
                        regs_a.commit(current_a, cid);
                    }
                    else
                    {
                        regs_b.commit(current_b, cid);
                    }
                }
                else
                {
                    regs_a.commit(current_a, tid);
                    regs_b.commit(current_b, tid);
                }
            }
        }

        A_tile_ptr += global_mult_A;
        B_tile_ptr += global_mult_B;
        __syncthreads();
    }

    if constexpr(use_direct_write)
    {
        auto store_wm = [&]<size_t wm>()
        {
            const int global_row = block_row + warp_m_base + wm * wmma_tile + half_warp_id;

            auto store_wn = [&]<size_t wn>()
            {
                const int global_col = block_col + warp_n_base + wn * wmma_tile + half_lane;
                store_matrix<LAYOUT_C, false>(C, c_frags[wm][wn], global_row, global_col, M, N);
            };

            [&]<size_t... wn>(std::index_sequence<wn...>)
            { (store_wn.template operator()<wn>(), ...); }(std::make_index_sequence<warp_tile_n>{});
        };

        [&]<size_t... wm>(std::index_sequence<wm...>)
        { (store_wm.template operator()<wm>(), ...); }(std::make_index_sequence<warp_tile_m>{});
    }
    else
    {
        // Calculate memory requirements
        constexpr int c_tile_bytes           = block_m * block_n * sizeof(T);
        constexpr int available_shared_bytes = 2 * lds_size * sizeof(U);

        constexpr bool is_col_major   = (LAYOUT_C == m_layout::col_major);
        constexpr int  chunk_elements = available_shared_bytes / sizeof(T);

        if constexpr(is_col_major)
        {
            // Chunk by columns - align to wmma_tile boundaries
            constexpr int raw_cols_per_chunk = chunk_elements / block_m;
            constexpr int cols_per_chunk     = (raw_cols_per_chunk / wmma_tile) * wmma_tile;
            constexpr int num_chunks         = (block_n + cols_per_chunk - 1) / cols_per_chunk;
            constexpr int last_chunk_cols    = block_n - ((num_chunks - 1) * cols_per_chunk);

            auto process_chunk = [&]<size_t ci>()
            {
                constexpr int col_start = static_cast<int>(ci) * cols_per_chunk;
                constexpr int chunk_cols
                    = (static_cast<int>(ci) == num_chunks - 1) ? last_chunk_cols : cols_per_chunk;

                T* c_chunk = reinterpret_cast<T*>(lds_mem);

                auto store_wm = [&]<size_t wm>()
                {
                    const int local_row
                        = warp_m_base + static_cast<int>(wm) * wmma_tile + half_warp_id;

                    auto store_wn = [&]<size_t wn>()
                    {
                        const int local_col
                            = warp_n_base + static_cast<int>(wn) * wmma_tile + half_lane;

                        if(local_col >= col_start && local_col < col_start + chunk_cols)
                        {
                            store_matrix<LAYOUT_C, true>(c_chunk,
                                                         c_frags[wm][wn],
                                                         local_row,
                                                         local_col - col_start,
                                                         block_m,
                                                         chunk_cols);
                        }
                    };

                    [&]<size_t... wn>(std::index_sequence<wn...>) {
                        (store_wn.template operator()<wn>(), ...);
                    }(std::make_index_sequence<warp_tile_n>{});
                };

                [&]<size_t... wm>(std::index_sequence<wm...>) {
                    (store_wm.template operator()<wm>(), ...);
                }(std::make_index_sequence<warp_tile_m>{});

                __syncthreads();

                load_shared_to_global<LAYOUT_C, bits, full_block, block_m, chunk_cols>(
                    C,
                    c_chunk,
                    block_row,
                    block_col + col_start,
                    M,
                    N,
                    tid);

                __syncthreads();
            };

            [&]<size_t... ci>(std::index_sequence<ci...>) {
                (process_chunk.template operator()<ci>(), ...);
            }(std::make_index_sequence<num_chunks>{});
        }
        else // row_major
        {
            // Chunk by rows - align to wmma_tile boundaries
            constexpr int raw_rows_per_chunk = chunk_elements / block_n;
            constexpr int rows_per_chunk     = (raw_rows_per_chunk / wmma_tile) * wmma_tile;
            constexpr int num_chunks         = (block_m + rows_per_chunk - 1) / rows_per_chunk;
            constexpr int last_chunk_rows    = block_m - ((num_chunks - 1) * rows_per_chunk);

            auto process_chunk = [&]<size_t ci>()
            {
                constexpr int row_start = static_cast<int>(ci) * rows_per_chunk;
                constexpr int chunk_rows
                    = (static_cast<int>(ci) == num_chunks - 1) ? last_chunk_rows : rows_per_chunk;

                T* c_chunk = reinterpret_cast<T*>(lds_mem);

                auto store_wm = [&]<size_t wm>()
                {
                    const int local_row
                        = warp_m_base + static_cast<int>(wm) * wmma_tile + half_warp_id;

                    if(local_row >= row_start && local_row < row_start + chunk_rows)
                    {
                        const int chunk_row = local_row - row_start;

                        auto store_wn = [&]<size_t wn>()
                        {
                            const int local_col
                                = warp_n_base + static_cast<int>(wn) * wmma_tile + half_lane;
                            store_matrix<LAYOUT_C, true>(c_chunk,
                                                         c_frags[wm][wn],
                                                         chunk_row,
                                                         local_col,
                                                         chunk_rows,
                                                         block_n);
                        };

                        [&]<size_t... wn>(std::index_sequence<wn...>) {
                            (store_wn.template operator()<wn>(), ...);
                        }(std::make_index_sequence<warp_tile_n>{});
                    }
                };

                [&]<size_t... wm>(std::index_sequence<wm...>) {
                    (store_wm.template operator()<wm>(), ...);
                }(std::make_index_sequence<warp_tile_m>{});

                __syncthreads();

                load_shared_to_global<LAYOUT_C, bits, full_block, chunk_rows, block_n>(
                    C,
                    c_chunk,
                    block_row + row_start,
                    block_col,
                    M,
                    N,
                    tid);

                __syncthreads();
            };

            [&]<size_t... ci>(std::index_sequence<ci...>) {
                (process_chunk.template operator()<ci>(), ...);
            }(std::make_index_sequence<num_chunks>{});
        }
    }
}

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
         bool     buffer_first,
         bool     use_async,
         bool     use_direct_write>
struct kernel_gemm_impl
{
    using config = wave_config<LAYOUT_A, LAYOUT_B, warps_m, warps_n>;

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
                  buffer_first,
                  use_async,
                  use_direct_write>(C, A, B, M, N, K);
    }
};

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_KERNEL_HPP
