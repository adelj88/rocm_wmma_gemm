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

#ifndef ROCM_WMMA_GEMM_KERNEL_LAUNCHER_HPP
#define ROCM_WMMA_GEMM_KERNEL_LAUNCHER_HPP

#include <rocm_wmma_gemm/kernel/config_generated.hpp>
#include <rocm_wmma_gemm/kernel_lookup.hpp>
#include <stdexcept>
#include <type_traits>

namespace rocm_wmma_gemm
{

template<class T, class U, m_layout layout_C, m_layout layout_A, m_layout layout_B>
struct kernel_launcher
{
    using kernel_func_ptr = void (*)(T*, const U*, const U*, int, int, int);

    static void launch(size_t       config_idx,
                       T*           C,
                       const U*     A,
                       const U*     B,
                       size_t       M,
                       size_t       N,
                       size_t       K,
                       dim3         grid_dim,
                       dim3         block_dim,
                       hipStream_t& stream)
    {
        // Compute type index (0-3)
        constexpr size_t type_idx
            = std::is_same_v<T, half> && std::is_same_v<U, half>                       ? 0
              : std::is_same_v<T, float> && std::is_same_v<U, half>                    ? 1
              : std::is_same_v<T, __hip_bfloat16> && std::is_same_v<U, __hip_bfloat16> ? 2
              : std::is_same_v<T, float> && std::is_same_v<U, __hip_bfloat16>          ? 3
                                                                                       : 0;

        // Compute layout index (0-7) based on (A,B,C) order to match lookup table
        constexpr size_t layout_idx
            = (layout_A == m_layout::row_major && layout_B == m_layout::row_major
               && layout_C == m_layout::row_major)
                  ? 0
              : (layout_A == m_layout::row_major && layout_B == m_layout::row_major
                 && layout_C == m_layout::col_major)
                  ? 1
              : (layout_A == m_layout::row_major && layout_B == m_layout::col_major
                 && layout_C == m_layout::row_major)
                  ? 2
              : (layout_A == m_layout::row_major && layout_B == m_layout::col_major
                 && layout_C == m_layout::col_major)
                  ? 3
              : (layout_A == m_layout::col_major && layout_B == m_layout::row_major
                 && layout_C == m_layout::row_major)
                  ? 4
              : (layout_A == m_layout::col_major && layout_B == m_layout::row_major
                 && layout_C == m_layout::col_major)
                  ? 5
              : (layout_A == m_layout::col_major && layout_B == m_layout::col_major
                 && layout_C == m_layout::row_major)
                  ? 6
              : (layout_A == m_layout::col_major && layout_B == m_layout::col_major
                 && layout_C == m_layout::col_major)
                  ? 7
                  : 0;

        // Lookup kernel function pointer from static table
        void* kernel_ptr = lookup_kernel(config_idx, type_idx, layout_idx);

        // Check for null pointer (kernel not available for this config/layout combination)
        if(kernel_ptr == nullptr)
        {
            // This should not happen if config selection is working correctly
            // Fall back to a default kernel or throw an error
            throw std::runtime_error(
                "No kernel available for the requested configuration and layout");
        }

        // Cast to correct function pointer type and launch
        auto kernel_func = reinterpret_cast<kernel_func_ptr>(kernel_ptr);
        kernel_func<<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
    }
};

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_KERNEL_LAUNCHER_HPP
