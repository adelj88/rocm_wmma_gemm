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

#ifndef ROCM_WMMA_GEMM_MAPPING_HPP
#define ROCM_WMMA_GEMM_MAPPING_HPP

namespace rocm_wmma_gemm
{

/**
 * @brief Hilbert curve tile mapping
 */
template<int BLOCK_M, int BLOCK_N>
class hilbert_mapping
{
private:
    static __device__ __forceinline__ int largest_power_of_2(int n)
    {
        constexpr int total_bits = sizeof(int) * 8;
        return 1 << ((total_bits - 1) - __clz(n));
    }

    static __device__ __forceinline__ bool is_power_of_2(int n)
    {
        return n > 0 && (n & (n - 1)) == 0;
    }

    static __device__ __forceinline__ void
        hilbert_coords(int local_tile_id, int hilbert_size, int* i, int* j)
    {
        *i    = 0;
        *j    = 0;
        int t = local_tile_id;

        for(int k = 1; k < hilbert_size; k <<= 1)
        {
            int bi = 1 & (t >> 1);
            int bj = 1 & (t ^ bi);
            if(bj == 0)
            {
                if(bi == 1)
                {
                    *i = k - 1 - *i; // flip up-down
                    *j = k - 1 - *j; // flip left-right
                }
                int tmp = *i; // transpose
                *i      = *j;
                *j      = tmp;
            }
            *i += k * bi;
            *j += k * bj;
            t >>= 2;
        }
    }

    static __device__ __forceinline__ void hilbert_recursive_mapping(int  tile_id,
                                                                     int  region_m,
                                                                     int  region_n,
                                                                     int  offset_m,
                                                                     int  offset_n,
                                                                     int* block_row,
                                                                     int* block_col)
    {
        // Base case: fall back to row-major for small or non-power-of-2 regions
        if(region_m == 1 || region_n == 1 || !is_power_of_2(region_m) || !is_power_of_2(region_n))
        {
            int row    = tile_id / region_n;
            int col    = tile_id % region_n;
            *block_row = (offset_m + row) * BLOCK_M;
            *block_col = (offset_n + col) * BLOCK_N;
            return;
        }

        // For power-of-2 regions, use Hilbert mapping
        const int hilbert_size     = min(region_m, region_n);
        const int tiles_per_square = hilbert_size * hilbert_size;

        // Calculate how many complete Hilbert squares fit
        const int squares_in_m         = region_m / hilbert_size;
        const int squares_in_n         = region_n / hilbert_size;
        const int total_complete_tiles = squares_in_m * squares_in_n * tiles_per_square;

        if(tile_id < total_complete_tiles)
        {
            // Main Hilbert region
            const int square_id     = tile_id / tiles_per_square;
            const int square_row    = square_id / squares_in_n;
            const int square_col    = square_id % squares_in_n;
            const int local_tile_id = tile_id % tiles_per_square;

            int i, j;
            hilbert_coords(local_tile_id, hilbert_size, &i, &j);

            *block_row = (offset_m + square_row * hilbert_size + i) * BLOCK_M;
            *block_col = (offset_n + square_col * hilbert_size + j) * BLOCK_N;
        }
        else
        {
            // Handle remainder regions recursively
            int remainder_id = tile_id - total_complete_tiles;

            const int remainder_m = region_m % hilbert_size;
            const int remainder_n = region_n % hilbert_size;

            // Right remainder strip (if region_n doesn't divide evenly)
            const int right_remainder_tiles = squares_in_m * hilbert_size * remainder_n;

            if(remainder_id < right_remainder_tiles && remainder_n > 0)
            {
                // Recursively handle right strip
                const int strip_row      = remainder_id / remainder_n;
                const int strip_local_id = remainder_id % remainder_n;

                hilbert_recursive_mapping(strip_local_id,
                                          1,
                                          remainder_n,
                                          offset_m + strip_row,
                                          offset_n + squares_in_n * hilbert_size,
                                          block_row,
                                          block_col);
            }
            else
            {
                // Bottom remainder region (including corner)
                remainder_id -= right_remainder_tiles;
                const int bottom_width = region_n; // Full width for bottom strip

                hilbert_recursive_mapping(remainder_id,
                                          remainder_m,
                                          bottom_width,
                                          offset_m + squares_in_m * hilbert_size,
                                          offset_n,
                                          block_row,
                                          block_col);
            }
        }
    }

public:
    static __device__ __forceinline__ void
        map_tile(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
    {
        hilbert_recursive_mapping(tile_id, grid_m, grid_n, 0, 0, block_row, block_col);
    }
};

/**
 * @brief Row-major tile mapping
 */
template<int BLOCK_M, int BLOCK_N>
class row_major_mapping
{
public:
    static __device__ __forceinline__ void
        map_tile(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
    {
        // Basic row-major calculation
        int row = tile_id / grid_n;
        int col = tile_id % grid_n;

        // Convert to actual block coordinates
        *block_row = row * BLOCK_M;
        *block_col = col * BLOCK_N;
    }
};

/**
 * @brief Column-major tile mapping
 */
template<int BLOCK_M, int BLOCK_N>
class col_major_mapping
{
public:
    static __device__ __forceinline__ void
        map_tile(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
    {
        // Basic column-major calculation
        int col = tile_id / grid_m;
        int row = tile_id % grid_m;

        // Convert to actual block coordinates
        *block_row = row * BLOCK_M;
        *block_col = col * BLOCK_N;
    }
};

/**
 * @brief Selector for tile mapping implementation based on layouts
 */
template<m_layout LAYOUT_A, m_layout LAYOUT_B>
struct select_tile_mapping_impl
{
    template<int BLOCK_M, int BLOCK_N>
    using type = hilbert_mapping<BLOCK_M, BLOCK_N>;
};

template<>
struct select_tile_mapping_impl<m_layout::row_major, m_layout::row_major>
{
    template<int BLOCK_M, int BLOCK_N>
    using type = row_major_mapping<BLOCK_M, BLOCK_N>;
};

template<int BLOCK_M, int BLOCK_N, m_layout LAYOUT_A, m_layout LAYOUT_B>
class tile_mapper
{
    using base_type =
        typename select_tile_mapping_impl<LAYOUT_A, LAYOUT_B>::template type<BLOCK_M, BLOCK_N>;

public:
    __device__ __forceinline__ void
        map_tile(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col) const
    {
        base_type::map_tile(tile_id, grid_m, grid_n, block_row, block_col);
    }
};

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_MAPPING_HPP
