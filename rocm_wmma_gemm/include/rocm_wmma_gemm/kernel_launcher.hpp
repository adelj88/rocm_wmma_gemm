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

#include "kernel/kernel.hpp"
#include <rocm_wmma_gemm/kernel/config_generated.hpp>

namespace rocm_wmma_gemm
{

namespace detail
{

// Helper to unpack tuple as template parameters
template<typename T,
         m_layout layout_C,
         m_layout layout_A,
         m_layout layout_B,
         typename Tuple,
         size_t... I>
__host__ void launch_kernel_impl(std::index_sequence<I...>,
                                 T*           C,
                                 const T*     A,
                                 const T*     B,
                                 size_t       M,
                                 size_t       N,
                                 size_t       K,
                                 dim3         grid_dim,
                                 dim3         block_dim,
                                 hipStream_t& stream)
{
    kernel_gemm<T,
                layout_C,
                layout_A,
                layout_B,
                std::get<0>(kernel_configs[I])...,
                std::get<1>(kernel_configs[I])...,
                std::get<2>(kernel_configs[I])...,
                std::get<3>(kernel_configs[I])...>
        <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

// Template to generate dispatch table
template<typename T, m_layout layout_C, m_layout layout_A, m_layout layout_B, size_t... I>
constexpr auto make_dispatch_table(std::index_sequence<I...>)
{
    using kernel_func
        = void (*)(T*, const T*, const T*, size_t, size_t, size_t, dim3, dim3, hipStream_t&);

    return std::array<kernel_func, sizeof...(I)>{
        [](T*           C,
           const T*     A,
           const T*     B,
           size_t       M,
           size_t       N,
           size_t       K,
           dim3         grid_dim,
           dim3         block_dim,
           hipStream_t& stream)
        {
            kernel_gemm<T,
                        layout_C,
                        layout_A,
                        layout_B,
                        std::get<0>(kernel_configs[I]),
                        std::get<1>(kernel_configs[I]),
                        std::get<2>(kernel_configs[I]),
                        std::get<3>(kernel_configs[I])>
                <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
        }...};
}

} // namespace detail

template<typename T, m_layout layout_C, m_layout layout_A, m_layout layout_B>
struct kernel_launcher
{
    static constexpr auto dispatch_table
        = detail::make_dispatch_table<T, layout_C, layout_A, layout_B>(
            std::make_index_sequence<KERNEL_VARIANTS>{});

    static void launch(const gemm_params& params,
                       T*                 C,
                       const T*           A,
                       const T*           B,
                       size_t             M,
                       size_t             N,
                       size_t             K,
                       dim3               grid_dim,
                       dim3               block_dim,
                       hipStream_t&       stream)
    {
        // Get index from params and dispatch to correct kernel
        size_t idx = get_kernel_config_index(params);
        dispatch_table[idx](C, A, B, M, N, K, grid_dim, block_dim, stream);
    }
};

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_KERNEL_LAUNCHER_HPP
