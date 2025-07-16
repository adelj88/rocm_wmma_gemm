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

#ifndef ROCM_WMMA_GEMM_COMMON_HPP
#define ROCM_WMMA_GEMM_COMMON_HPP

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace rocm_wmma_gemm
{

/**
 * @brief Enum class defining matrix layout options
 */
enum class m_layout
{
    row_major, ///< Row-major layout (elements consecutive in memory by row)
    col_major ///< Column-major layout (elements consecutive in memory by column)
};

// Enum to specify which matrix is being accessed (A or B)
enum class m_input
{
    matrix_a,
    matrix_b
};

constexpr int wmma_tile = 16;
constexpr int warp_size = 32;

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_COMMON_HPP
