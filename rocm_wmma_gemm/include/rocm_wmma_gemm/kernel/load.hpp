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

#ifndef ROCM_WMMA_GEMM_LOAD_HPP
#define ROCM_WMMA_GEMM_LOAD_HPP

namespace rocm_wmma_gemm
{

// Col-major load with optional padding
template<m_layout ACCESS,
         int      MAX_BITS,
         int      BLOCK_SIZE,
         int      BLOCK_M,
         int      BLOCK_N,
         int      PADDING,
         class T>
__device__ __forceinline__ auto load_to_shared(T* output, const T* input, int M, int N, int tid) ->
    typename std::enable_if<ACCESS == m_layout::col_major, void>::type
{
    // For col-major, padding is added to the row dimension (leading dimension)
    constexpr int padded_rows = BLOCK_M + PADDING;

    constexpr int min_block_dim   = (BLOCK_M < BLOCK_N) ? BLOCK_M : BLOCK_N;
    constexpr int min_block_bytes = min_block_dim * sizeof(T);

    constexpr int element_alignment = min_block_bytes / sizeof(T);
    constexpr int calculated_width  = element_alignment & (-element_alignment);
    constexpr int max_bytes         = MAX_BITS / 8;
    constexpr int max_vector_width  = max_bytes / sizeof(T);
    constexpr int actual_load_width
        = (calculated_width > max_vector_width) ? max_vector_width : calculated_width;

    using type                 = typename type_selector<T>::type;
    using vector_type          = type __attribute__((ext_vector_type(actual_load_width)));
    constexpr int vector_width = (sizeof(vector_type) / sizeof(T));

    // Total elements to load (without padding)
    constexpr int total_elements = BLOCK_M * BLOCK_N;
    constexpr int vectors_per_thread
        = ((total_elements / vector_width) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int i = 0; i < vectors_per_thread; ++i)
    {
        const int idx = (tid * vector_width) + (i * BLOCK_SIZE * vector_width);

        if(idx < total_elements)
        {
            // Compute position in logical (no padding) layout
            const int col = idx / BLOCK_M;
            const int row = idx % BLOCK_M;

            // Global memory index (no padding)
            const int gload = col * M + row;

            // Shared memory index (with padding in row dimension)
            const int sstore = col * padded_rows + row;

            *reinterpret_cast<vector_type*>(output + sstore)
                = *reinterpret_cast<const vector_type*>(input + gload);
        }
    }
}

// Row-major load with optional padding
template<m_layout ACCESS,
         int      MAX_BITS,
         int      BLOCK_SIZE,
         int      BLOCK_M,
         int      BLOCK_N,
         int      PADDING,
         class T>
__device__ __forceinline__ auto load_to_shared(T* output, const T* input, int M, int N, int tid) ->
    typename std::enable_if<ACCESS == m_layout::row_major, void>::type
{
    // For row-major, padding is added to the column dimension (leading dimension)
    constexpr int padded_cols = BLOCK_N + PADDING;

    constexpr int min_block_dim   = (BLOCK_M < BLOCK_N) ? BLOCK_M : BLOCK_N;
    constexpr int min_block_bytes = min_block_dim * sizeof(T);

    constexpr int element_alignment = min_block_bytes / sizeof(T);
    constexpr int calculated_width  = element_alignment & (-element_alignment);
    constexpr int max_bytes         = MAX_BITS / 8;
    constexpr int max_vector_width  = max_bytes / sizeof(T);
    constexpr int actual_load_width
        = (calculated_width > max_vector_width) ? max_vector_width : calculated_width;

    using type                 = typename type_selector<T>::type;
    using vector_type          = type __attribute__((ext_vector_type(actual_load_width)));
    constexpr int vector_width = (sizeof(vector_type) / sizeof(T));

    // Total elements to load (without padding)
    constexpr int total_elements = BLOCK_M * BLOCK_N;
    constexpr int vectors_per_thread
        = ((total_elements / vector_width) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int i = 0; i < vectors_per_thread; ++i)
    {
        const int idx = (tid * vector_width) + (i * BLOCK_SIZE * vector_width);

        if(idx < total_elements)
        {
            // Compute position in logical (no padding) layout
            const int row = idx / BLOCK_N;
            const int col = idx % BLOCK_N;

            // Global memory index (no padding)
            const int gload = row * N + col;

            // Shared memory index (with padding in column dimension)
            const int sstore = row * padded_cols + col;

            *reinterpret_cast<vector_type*>(output + sstore)
                = *reinterpret_cast<const vector_type*>(input + gload);
        }
    }
}

template<m_layout ACCESS, int MAX_BITS, int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto
    load_shared_to_global(T* output, T* input, int row, int col, int M, int N, int tid) ->
    typename std::enable_if<ACCESS == m_layout::col_major, void>::type
{
    constexpr int min_block_dim   = (BLOCK_M < BLOCK_N) ? BLOCK_M : BLOCK_N;
    constexpr int min_block_bytes = min_block_dim * sizeof(T);

    constexpr int element_alignment = min_block_bytes / sizeof(T);
    constexpr int calculated_width  = element_alignment & (-element_alignment);
    constexpr int max_bytes         = MAX_BITS / 8;
    constexpr int max_vector_width  = max_bytes / sizeof(T);
    constexpr int actual_load_width
        = (calculated_width > max_vector_width) ? max_vector_width : calculated_width;

    using type                 = typename type_selector<T>::type;
    using vector_type          = type __attribute__((ext_vector_type(actual_load_width)));
    constexpr int vector_width = (sizeof(vector_type) / sizeof(T));
    constexpr int vectors_per_thread
        = (((BLOCK_M * BLOCK_N) / vector_width) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int i = 0; i < vectors_per_thread; ++i)
    {
        const int idx = (tid * vector_width) + (i * BLOCK_SIZE * vector_width);

        if(idx < (BLOCK_M * BLOCK_N))
        {
            const int local_col = idx / BLOCK_M;
            const int local_row = idx % BLOCK_M;

            const int global_row = row + local_row;
            const int global_col = col + local_col;

            // Check if this vector is entirely within bounds (down the column)
            if(global_col < N && (global_row + vector_width - 1) < M)
            {
                // Full vector write
                *reinterpret_cast<vector_type*>(output + global_col * M + global_row)
                    = *reinterpret_cast<const vector_type*>(input + idx);
            }
            else if(global_col < N)
            {
                // Handle boundary case element by element
                for(int v = 0; v < vector_width; v++)
                {
                    if((global_row + v) < M)
                    {
                        output[global_col * M + global_row + v] = input[idx + v];
                    }
                }
            }
        }
    }
}

template<m_layout ACCESS, int MAX_BITS, int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto
    load_shared_to_global(T* output, T* input, int row, int col, int M, int N, int tid) ->
    typename std::enable_if<ACCESS == m_layout::row_major, void>::type
{
    constexpr int min_block_dim   = (BLOCK_M < BLOCK_N) ? BLOCK_M : BLOCK_N;
    constexpr int min_block_bytes = min_block_dim * sizeof(T);

    constexpr int element_alignment = min_block_bytes / sizeof(T);
    constexpr int calculated_width  = element_alignment & (-element_alignment);
    constexpr int max_bytes         = MAX_BITS / 8;
    constexpr int max_vector_width  = max_bytes / sizeof(T);
    constexpr int actual_load_width
        = (calculated_width > max_vector_width) ? max_vector_width : calculated_width;

    using type                 = typename type_selector<T>::type;
    using vector_type          = type __attribute__((ext_vector_type(actual_load_width)));
    constexpr int vector_width = (sizeof(vector_type) / sizeof(T));
    constexpr int vectors_per_thread
        = (((BLOCK_M * BLOCK_N) / vector_width) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int i = 0; i < vectors_per_thread; ++i)
    {
        const int idx = (tid * vector_width) + (i * BLOCK_SIZE * vector_width);

        if(idx < (BLOCK_M * BLOCK_N))
        {
            const int local_row = idx / BLOCK_N;
            const int local_col = idx % BLOCK_N;

            const int global_row = row + local_row;
            const int global_col = col + local_col;

            // Check if this vector is entirely within bounds
            if(global_row < M && (global_col + vector_width - 1) < N)
            {
                // Full vector write
                *reinterpret_cast<vector_type*>(output + global_row * N + global_col)
                    = *reinterpret_cast<const vector_type*>(input + idx);
            }
            else if(global_row < M)
            {
                // Handle boundary case element by element
                for(int v = 0; v < vector_width; v++)
                {
                    if((global_col + v) < N)
                    {
                        output[global_row * N + global_col + v] = input[idx + v];
                    }
                }
            }
        }
    }
}

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_LOAD_HPP
