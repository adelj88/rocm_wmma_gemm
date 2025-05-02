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

#ifndef ROCM_WMMA_GEMM_WMMA_HPP
#define ROCM_WMMA_GEMM_WMMA_HPP

namespace rocm_wmma_gemm
{

template<class T1, class T2, int TILE>
__device__ __forceinline__ auto
    wmma(fragment<T1, TILE>& frag1, fragment<T1, TILE>& frag2, fragment<T2, TILE>& frag3) ->
    typename std::enable_if<(std::is_same<T1, half>::value && std::is_same<T2, half>::value),
                            void>::type
{
    frag3.get()
        = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(frag1.get(), frag2.get(), frag3.get(), false);
}

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_WMMA_HPP
