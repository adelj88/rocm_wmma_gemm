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

template<m_input MATRIX, m_layout ACCESS, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto
    load_to_shared(T* output, const T* input, int M, int N, int tid, int block_size) ->
    typename std::enable_if<MATRIX == m_input::matrix_a && ACCESS == m_layout::row_major,
                            void>::type
{
    using vector_type                      = float __attribute__((ext_vector_type(16)));
    using half_vector_type                 = float __attribute__((ext_vector_type(8)));
    static constexpr int vector_width      = (sizeof(vector_type) / sizeof(T));
    static constexpr int half_vector_width = (sizeof(half_vector_type) / sizeof(T));

    for(int i = tid * vector_width; i < (BLOCK_M * BLOCK_N); i += block_size * vector_width)
    {
        const int i0 = i;
        const int i1 = i + half_vector_width;

        const int row0   = i0 / BLOCK_N;
        const int col0   = i0 % BLOCK_N;
        const int gload0 = row0 * N + col0;

        const int row1   = i1 / BLOCK_N;
        const int col1   = i1 % BLOCK_N;
        const int gload1 = row1 * N + col1;

        *reinterpret_cast<half_vector_type*>(output + i0)
            = *reinterpret_cast<const half_vector_type*>(input + gload0);
        *reinterpret_cast<half_vector_type*>(output + i1)
            = *reinterpret_cast<const half_vector_type*>(input + gload1);
    }
}

template<m_input MATRIX, m_layout ACCESS, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto
    load_to_shared(T* output, const T* input, int M, int N, int tid, int block_size) ->
    typename std::enable_if<MATRIX == m_input::matrix_b && ACCESS == m_layout::col_major,
                            void>::type
{
    using vector_type                      = float __attribute__((ext_vector_type(16)));
    using half_vector_type                 = float __attribute__((ext_vector_type(8)));
    static constexpr int vector_width      = (sizeof(vector_type) / sizeof(T));
    static constexpr int half_vector_width = (sizeof(half_vector_type) / sizeof(T));

    for(int i = tid * vector_width; i < (BLOCK_M * BLOCK_N); i += block_size * vector_width)
    {
        const int i0 = i;
        const int i1 = i + half_vector_width;

        const int col0   = i0 / BLOCK_M;
        const int row0   = i0 % BLOCK_M;
        const int gload0 = col0 * M + row0;

        const int col1   = i1 / BLOCK_M;
        const int row1   = i1 % BLOCK_M;
        const int gload1 = col1 * M + row1;

        *reinterpret_cast<half_vector_type*>(output + i0)
            = *reinterpret_cast<const half_vector_type*>(input + gload0);
        *reinterpret_cast<half_vector_type*>(output + i1)
            = *reinterpret_cast<const half_vector_type*>(input + gload1);
    }
}

template<m_input MATRIX, m_layout ACCESS, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto
    load_to_shared(T* output, const T* input, int M, int N, int tid, int block_size) ->
    typename std::enable_if<MATRIX == m_input::matrix_a && ACCESS == m_layout::col_major,
                            void>::type
{
    using vector_type                 = float __attribute__((ext_vector_type(16)));
    static constexpr int vector_width = (sizeof(vector_type) / sizeof(T));

    for(int i = tid * vector_width; i < (BLOCK_M * BLOCK_N); i += block_size * vector_width)
    {
        const int col   = i / BLOCK_M;
        const int row   = i % BLOCK_M;
        int       gload = col * M + row;

        // Load full vector (buffer_load_b128 should handle out-of-bound accesses)
        *reinterpret_cast<vector_type*>(output + i)
            = *reinterpret_cast<const vector_type*>(input + gload);
    }
}

template<m_input MATRIX, m_layout ACCESS, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto
    load_to_shared(T* output, const T* input, int M, int N, int tid, int block_size) ->
    typename std::enable_if<MATRIX == m_input::matrix_b && ACCESS == m_layout::row_major,
                            void>::type
{
    using vector_type                 = float __attribute__((ext_vector_type(16)));
    static constexpr int vector_width = (sizeof(vector_type) / sizeof(T));

    for(int i = tid * vector_width; i < (BLOCK_M * BLOCK_N); i += block_size * vector_width)
    {
        const int row   = i / BLOCK_N;
        const int col   = i % BLOCK_N;
        int       gload = row * N + col;

        // Load full vector (buffer_load_b128 should handle out-of-bound accesses)
        *reinterpret_cast<vector_type*>(output + i)
            = *reinterpret_cast<const vector_type*>(input + gload);
    }
}

template<m_layout ACCESS, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto load_shared_to_global(
    T* output, T* input, int row, int col, int M, int N, int tid, int block_size) ->
    typename std::enable_if<ACCESS == m_layout::col_major, void>::type
{
    using vector_type                 = float __attribute__((ext_vector_type(16)));
    static constexpr int vector_width = (sizeof(vector_type) / sizeof(T));

    for(int i = tid * vector_width; i < (BLOCK_M * BLOCK_N); i += block_size * vector_width)
    {
        const int local_col = i / BLOCK_M;
        const int local_row = i % BLOCK_M;

        const int global_row = row + local_row;
        const int global_col = col + local_col;

        // Check if this vector is entirely within bounds (down the column)
        if(global_col < N && (global_row + vector_width - 1) < M)
        {
            // Full vector write
            *reinterpret_cast<vector_type*>(output + global_col * M + global_row)
                = *reinterpret_cast<const vector_type*>(input + i);
        }
        else if(global_col < N)
        {
            // Handle boundary case element by element
            for(int v = 0; v < vector_width; v++)
            {
                if((global_row + v) < M)
                {
                    output[global_col * M + global_row + v] = input[i + v];
                }
            }
        }
    }
}

template<m_layout ACCESS, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto load_shared_to_global(
    T* output, T* input, int row, int col, int M, int N, int tid, int block_size) ->
    typename std::enable_if<ACCESS == m_layout::row_major, void>::type
{
    using vector_type                 = float __attribute__((ext_vector_type(16)));
    static constexpr int vector_width = (sizeof(vector_type) / sizeof(T));

    for(int i = tid * vector_width; i < (BLOCK_M * BLOCK_N); i += block_size * vector_width)
    {
        const int local_row = i / BLOCK_N;
        const int local_col = i % BLOCK_N;

        const int global_row = row + local_row;
        const int global_col = col + local_col;

        // Check if this vector is entirely within bounds
        if(global_row < M && global_col + vector_width - 1 < N)
        {
            // Full vector write
            *reinterpret_cast<vector_type*>(output + global_row * N + global_col)
                = *reinterpret_cast<const vector_type*>(input + i);
        }
        else if(global_row < M)
        {
            // Handle boundary case element by element
            for(int v = 0; v < vector_width; v++)
            {
                if((global_col + v) < N)
                {
                    output[global_row * N + global_col + v] = input[i + v];
                }
            }
        }
    }
}

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_LOAD_HPP
