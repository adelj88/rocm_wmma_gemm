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

#ifndef ROCM_WMMA_GEMM_HPP
#define ROCM_WMMA_GEMM_HPP

#include "kernel/common.hpp"

namespace rocm_wmma_gemm
{

// Main GEMM function - types at the end for inference
template<m_layout layout_C, m_layout layout_A, m_layout layout_B, class T, class U>
__host__ void gemm(T* C, U* A, U* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template<m_layout layout_C, m_layout layout_A, m_layout layout_B, class T, class U>
__host__ void
    gemm(T* C, U* A, U* B, size_t M, size_t N, size_t K, size_t batch_count, hipStream_t& stream);

// Macro to declare all layout combinations for a type pair
#define DECLARE_GEMM_FOR_TYPES(T, U)                                                          \
    extern template void gemm<m_layout::row_major, m_layout::row_major, m_layout::row_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::row_major, m_layout::row_major, m_layout::col_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::row_major, m_layout::col_major, m_layout::row_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::row_major, m_layout::col_major, m_layout::col_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::col_major, m_layout::row_major, m_layout::row_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::col_major, m_layout::row_major, m_layout::col_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::col_major, m_layout::col_major, m_layout::row_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::col_major, m_layout::col_major, m_layout::col_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::row_major, m_layout::row_major, m_layout::row_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::row_major, m_layout::row_major, m_layout::col_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::row_major, m_layout::col_major, m_layout::row_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::row_major, m_layout::col_major, m_layout::col_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::col_major, m_layout::row_major, m_layout::row_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::col_major, m_layout::row_major, m_layout::col_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::col_major, m_layout::col_major, m_layout::row_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::col_major, m_layout::col_major, m_layout::col_major>( \
        T*,                                                                                   \
        U*,                                                                                   \
        U*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);

DECLARE_GEMM_FOR_TYPES(half, half)
DECLARE_GEMM_FOR_TYPES(float, half)
DECLARE_GEMM_FOR_TYPES(__hip_bfloat16, __hip_bfloat16)
DECLARE_GEMM_FOR_TYPES(float, __hip_bfloat16)

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_HPP
