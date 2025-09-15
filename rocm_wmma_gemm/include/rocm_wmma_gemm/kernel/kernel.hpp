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

template<class T,
         class U,
         m_layout LAYOUT_C,
         m_layout LAYOUT_A,
         m_layout LAYOUT_B,
         int      warps_m,
         int      warps_n,
         int      warp_tile_m,
         int      warp_tile_n,
         int      bits,
         bool     use_direct_write>
__global__ __launch_bounds__(warp_size* warps_m* warps_n) void kernel_gemm(
    T* C, const U* A, const U* B, int M, int N, int K)
{
    constexpr int block_m  = warps_m * warp_tile_m * wmma_tile; // 4*4*16 = bits
    constexpr int block_n  = warps_n * warp_tile_n * wmma_tile; // 4*4*16 = bits
    constexpr int block_k  = wmma_tile;
    constexpr int lds_size = (block_m * block_k) + (block_k * block_n);

    // Calculate grid dimensions
    const int grid_m  = (M + block_m - 1) / block_m;
    const int grid_n  = (N + block_n - 1) / block_n;
    const int tile_id = blockIdx.x;

    constexpr bool use_row = (LAYOUT_A == m_layout::row_major && LAYOUT_B == m_layout::row_major);
    constexpr bool use_col = (LAYOUT_A == m_layout::col_major && LAYOUT_B == m_layout::col_major);
    constexpr bool use_hilbert = !use_row && !use_col;

    using mapper = tile_mapper<block_m, block_n, LAYOUT_A, LAYOUT_B>;

    // Get block coordinates
    int block_row, block_col;
    mapper().map_tile(tile_id, grid_m, grid_n, &block_row, &block_col);

    // Allocate a unified shared memory buffer.
    __shared__ U lds_mem[2 * lds_size];

    // Partition the shared memory with manual offset calculations:
    // A tiles occupy the first region in each buffer
    U* a_tiles_0 = lds_mem;
    U* a_tiles_1 = lds_mem + lds_size;
    // B tiles start after A's region in each buffer
    U* b_tiles_0 = lds_mem + (block_m * block_k);
    U* b_tiles_1 = lds_mem + lds_size + (block_m * block_k);

    // Each block is launched with a one-dimensional thread block.
    constexpr int full_block = warp_size * warps_m * warps_n;
    constexpr int half_block = full_block / 2;
    const int     tid        = threadIdx.x;

    const int cid = tid % half_block;

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

    // Base pointers for the current A and B tiles.
    const U* A_tile_ptr = A_base;
    const U* B_tile_ptr = B_base;

    if(tid < half_block)
    {
        load_to_shared<LAYOUT_A, bits, half_block, block_m, block_k>(a_tiles_0,
                                                                     A_tile_ptr,
                                                                     M,
                                                                     K,
                                                                     cid);
    }
    else
    {
        load_to_shared<LAYOUT_B, bits, half_block, block_k, block_n>(b_tiles_0,
                                                                     B_tile_ptr,
                                                                     K,
                                                                     N,
                                                                     cid);
    }
    __syncthreads();

    U* current_a = a_tiles_0;
    U* current_b = b_tiles_0;
    U* next_a    = a_tiles_1;
    U* next_b    = b_tiles_1;

    const int global_mult_A = block_k * ((LAYOUT_A == m_layout::col_major) ? M : 1);
    const int global_mult_B = block_k * ((LAYOUT_B == m_layout::col_major) ? 1 : N);

    constexpr int frag_mult_A = (LAYOUT_A == m_layout::col_major) ? 1 : block_k;
    constexpr int frag_mult_B = (LAYOUT_B == m_layout::col_major) ? block_k : 1;

    constexpr int frag_offset_A = wmma_tile * frag_mult_A;
    constexpr int frag_offset_B = wmma_tile * frag_mult_B;

    const int warp_offset_A = (warp_m_base + half_lane) * frag_mult_A;
    const int warp_offset_B = (warp_n_base + half_lane) * frag_mult_B;

    // Main loop over k-dimension
    for(int k_tile = 0; k_tile < K; k_tile += block_k)
    {
        if(k_tile + block_k < K)
        {
            if(tid < half_block)
            {
                const U* next_A = A_tile_ptr + global_mult_A;
                load_to_shared<LAYOUT_A, bits, half_block, block_m, block_k>(next_a,
                                                                             next_A,
                                                                             M,
                                                                             K,
                                                                             cid);
            }
            else
            {
                const U* next_B = B_tile_ptr + global_mult_B;
                load_to_shared<LAYOUT_B, bits, half_block, block_k, block_n>(next_b,
                                                                             next_B,
                                                                             K,
                                                                             N,
                                                                             cid);
            }
        }

        const U* curr_a = current_a + warp_offset_A;
        const U* curr_b = current_b + warp_offset_B;

        if constexpr(warp_tile_m < warp_tile_n)
        {
            for(int wm = 0; wm < warp_tile_n; ++wm)
            {
                if(wm < warp_tile_m)
                {
                    load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[wm], curr_a, block_m, block_k);
                    curr_a += frag_offset_A;
                }
                load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[wm], curr_b, block_k, block_n);
                curr_b += frag_offset_B;
            }
        }
        else if constexpr(warp_tile_m > warp_tile_n)
        {
            for(int wm = 0; wm < warp_tile_m; ++wm)
            {
                load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[wm], curr_a, block_m, block_k);
                if(wm < warp_tile_n)
                {
                    load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[wm], curr_b, block_k, block_n);
                    curr_b += frag_offset_B;
                }
                curr_a += frag_offset_A;
            }
        }
        else if constexpr(warp_tile_m == warp_tile_n)
        {
            for(int wm = 0; wm < warp_tile_m; ++wm)
            {
                load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[wm], curr_a, block_m, block_k);
                load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[wm], curr_b, block_k, block_n);
                curr_a += frag_offset_A;
                curr_b += frag_offset_B;
            }
        }

        // Compute: each warp performs WMMA on its fragments.
        for(int wm = 0; wm < warp_tile_m; ++wm)
        {
            for(int wn = 0; wn < warp_tile_n; ++wn)
            {
                wmma(a_frag[wm], b_frag[wn], c_frags[wm][wn]);
            }
        }

        // Advance the global pointers for A and B tiles.
        A_tile_ptr += global_mult_A;
        B_tile_ptr += global_mult_B;
        U* temp_a = current_a;
        U* temp_b = current_b;
        current_a = next_a;
        current_b = next_b;
        next_a    = temp_a;
        next_b    = temp_b;
        __syncthreads();
    }

    if constexpr(use_direct_write)
    {
        // Direct write to global memory
        for(int wm = 0; wm < warp_tile_m; ++wm)
        {
            const int global_row = block_row + warp_m_base + wm * wmma_tile + half_warp_id;

            for(int wn = 0; wn < warp_tile_n; ++wn)
            {
                const int global_col = block_col + warp_n_base + wn * wmma_tile + half_lane;

                store_matrix<LAYOUT_C, false>(C, c_frags[wm][wn], global_row, global_col, M, N);
            }
        }
    }
    else
    {
        // Calculate memory requirements
        constexpr int c_tile_elements        = block_m * block_n;
        constexpr int c_tile_bytes           = c_tile_elements * sizeof(T);
        constexpr int available_shared_bytes = 2 * lds_size * sizeof(U);

        // Complex case: C tile doesn't fit, need chunking
        constexpr bool is_col_major   = (LAYOUT_C == m_layout::col_major);
        constexpr int  chunk_elements = available_shared_bytes / sizeof(T);

        if constexpr(is_col_major)
        {
            // Chunk by columns - align to wmma_tile boundaries
            constexpr int raw_cols_per_chunk = chunk_elements / block_m;
            constexpr int cols_per_chunk     = (raw_cols_per_chunk / wmma_tile) * wmma_tile;
            constexpr int num_chunks         = (block_n + cols_per_chunk - 1) / cols_per_chunk;
            constexpr int last_chunk_cols    = block_n - ((num_chunks - 1) * cols_per_chunk);

            for(int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
            {
                const int col_start = chunk_idx * cols_per_chunk;

                T* c_chunk = reinterpret_cast<T*>(lds_mem);

                // Store fragments to shared memory
                for(int wm = 0; wm < warp_tile_m; ++wm)
                {
                    const int local_row = warp_m_base + wm * wmma_tile + half_warp_id;

                    for(int wn = 0; wn < warp_tile_n; ++wn)
                    {
                        const int local_col = warp_n_base + wn * wmma_tile + half_lane;

                        // Only store if this fragment intersects with current chunk
                        const int chunk_cols
                            = (chunk_idx == num_chunks - 1) ? last_chunk_cols : cols_per_chunk;
                        if(local_col >= col_start && local_col < col_start + chunk_cols)
                        {
                            const int chunk_col = local_col - col_start;
                            store_matrix<LAYOUT_C, true>(c_chunk,
                                                         c_frags[wm][wn],
                                                         local_row,
                                                         chunk_col,
                                                         block_m,
                                                         chunk_cols);
                        }
                    }
                }
                __syncthreads();

                // Copy chunk to global memory - compile-time dispatch
                if constexpr(num_chunks == 1)
                {
                    // Only one chunk - use actual size
                    load_shared_to_global<LAYOUT_C, bits, full_block, block_m, last_chunk_cols>(
                        C,
                        c_chunk,
                        block_row,
                        block_col,
                        M,
                        N,
                        tid);
                }
                else
                {
                    // Multiple chunks - dispatch based on chunk index
                    if(chunk_idx < num_chunks - 1)
                    {
                        // Regular chunk
                        load_shared_to_global<LAYOUT_C, bits, full_block, block_m, cols_per_chunk>(
                            C,
                            c_chunk,
                            block_row,
                            block_col + col_start,
                            M,
                            N,
                            tid);
                    }
                    else
                    {
                        // Last chunk with different size
                        load_shared_to_global<LAYOUT_C, bits, full_block, block_m, last_chunk_cols>(
                            C,
                            c_chunk,
                            block_row,
                            block_col + col_start,
                            M,
                            N,
                            tid);
                    }
                }
                __syncthreads();
            }
        }
        else // row_major
        {
            // Chunk by rows - align to wmma_tile boundaries
            constexpr int raw_rows_per_chunk = chunk_elements / block_n;
            constexpr int rows_per_chunk     = (raw_rows_per_chunk / wmma_tile) * wmma_tile;
            constexpr int num_chunks         = (block_m + rows_per_chunk - 1) / rows_per_chunk;
            constexpr int last_chunk_rows    = block_m - ((num_chunks - 1) * rows_per_chunk);

            for(int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
            {
                const int row_start = chunk_idx * rows_per_chunk;

                T* c_chunk = reinterpret_cast<T*>(lds_mem);

                // Store fragments to shared memory
                for(int wm = 0; wm < warp_tile_m; ++wm)
                {
                    const int local_row = warp_m_base + wm * wmma_tile + half_warp_id;

                    // Only store if this fragment intersects with current chunk
                    const int chunk_rows
                        = (chunk_idx == num_chunks - 1) ? last_chunk_rows : rows_per_chunk;
                    if(local_row >= row_start && local_row < row_start + chunk_rows)
                    {
                        const int chunk_row = local_row - row_start;

                        for(int wn = 0; wn < warp_tile_n; ++wn)
                        {
                            const int local_col = warp_n_base + wn * wmma_tile + half_lane;
                            store_matrix<LAYOUT_C, true>(c_chunk,
                                                         c_frags[wm][wn],
                                                         chunk_row,
                                                         local_col,
                                                         chunk_rows,
                                                         block_n);
                        }
                    }
                }
                __syncthreads();

                // Copy chunk to global memory - compile-time dispatch
                if constexpr(num_chunks == 1)
                {
                    // Only one chunk - use actual size
                    load_shared_to_global<LAYOUT_C, bits, full_block, last_chunk_rows, block_n>(
                        C,
                        c_chunk,
                        block_row,
                        block_col,
                        M,
                        N,
                        tid);
                }
                else
                {
                    // Multiple chunks - dispatch based on chunk index
                    if(chunk_idx < num_chunks - 1)
                    {
                        // Regular chunk
                        load_shared_to_global<LAYOUT_C, bits, full_block, rows_per_chunk, block_n>(
                            C,
                            c_chunk,
                            block_row + row_start,
                            block_col,
                            M,
                            N,
                            tid);
                    }
                    else
                    {
                        // Last chunk with different size
                        load_shared_to_global<LAYOUT_C, bits, full_block, last_chunk_rows, block_n>(
                            C,
                            c_chunk,
                            block_row + row_start,
                            block_col,
                            M,
                            N,
                            tid);
                    }
                }
                __syncthreads();
            }
        }
    }
}

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_KERNEL_HPP
