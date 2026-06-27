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

#ifndef ROCM_WMMA_GEMM_KERNEL_LOADER_HPP
#define ROCM_WMMA_GEMM_KERNEL_LOADER_HPP

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <rocm_wmma_gemm/kernel/common.hpp>
#include <stdexcept>
#include <string>

namespace rocm_wmma_gemm
{

/**
 * @brief Runtime arch-library loader.
 *
 * Detects the current GPU architecture, opens the matching per-arch shared
 * library (librocm_wmma_gemm_<arch>.so), and exposes a gemm() call that
 * dispatches through the library's C ABI entry points. This allows a single
 * application binary to support multiple GPU architectures without recompiling,
 * simply by having the appropriate arch library present at runtime.
 *
 * Usage:
 *   rocm_wmma_gemm::loader loader;
 *   loader.gemm<m_layout::row_major, m_layout::row_major, m_layout::row_major>(
 *       C, A, B, M, N, K, stream);
 */
class loader
{
    using dispatch_fn
        = void (*)(void*, void*, void*, size_t, size_t, size_t, size_t, int, int, int, void*);

    void*       handle_       = nullptr;
    dispatch_fn fn_f16_f16_   = nullptr;
    dispatch_fn fn_f32_f16_   = nullptr;
    dispatch_fn fn_bf16_bf16_ = nullptr;
    dispatch_fn fn_f32_bf16_  = nullptr;

    static dispatch_fn resolve(void* handle, const char* name)
    {
        auto fn = reinterpret_cast<dispatch_fn>(dlsym(handle, name));
        if(!fn)
        {
            throw std::runtime_error(std::string("rocm_wmma_gemm::loader: symbol not found: ")
                                     + name + " (" + dlerror() + ")");
        }
        return fn;
    }

public:
    loader()
    {
        // Detect the current GPU architecture.
        hipDeviceProp_t props{};
        if(hipGetDeviceProperties(&props, 0) != hipSuccess)
        {
            throw std::runtime_error("rocm_wmma_gemm::loader: hipGetDeviceProperties failed");
        }

        // Strip feature flags (e.g. "gfx1100:xnack-" -> "gfx1100").
        std::string arch  = props.gcnArchName;
        const auto  colon = arch.find(':');
        if(colon != std::string::npos)
        {
            arch = arch.substr(0, colon);
        }

        const std::string libname = "librocm_wmma_gemm_" + arch + ".so";

        // Try the build-time known library directory first (injected by CMake),
        // then fall back to the system linker search path.
#ifdef ROCM_WMMA_GEMM_LIB_DIR
        handle_ = dlopen((std::string(ROCM_WMMA_GEMM_LIB_DIR "/") + libname).c_str(),
                         RTLD_NOW | RTLD_LOCAL);
#endif
        if(!handle_)
        {
            handle_ = dlopen(libname.c_str(), RTLD_NOW | RTLD_LOCAL);
        }

        if(!handle_)
        {
            throw std::runtime_error("rocm_wmma_gemm::loader: could not open " + libname + ": "
                                     + dlerror());
        }

        fn_f16_f16_   = resolve(handle_, "rocm_wmma_gemm_f16_f16");
        fn_f32_f16_   = resolve(handle_, "rocm_wmma_gemm_f32_f16");
        fn_bf16_bf16_ = resolve(handle_, "rocm_wmma_gemm_bf16_bf16");
        fn_f32_bf16_  = resolve(handle_, "rocm_wmma_gemm_f32_bf16");
    }

    ~loader()
    {
        if(handle_)
        {
            dlclose(handle_);
        }
    }

    loader(const loader&)            = delete;
    loader& operator=(const loader&) = delete;
    loader(loader&&)                 = delete;
    loader& operator=(loader&&)      = delete;

    /**
     * @brief Execute a GEMM via the arch-specific library.
     */
    template<m_layout layout_C, m_layout layout_A, m_layout layout_B, class T, class U>
    void gemm(T* C, U* A, U* B, size_t M, size_t N, size_t K, hipStream_t& stream)
    {
        gemm<layout_C, layout_A, layout_B>(C, A, B, M, N, K, 1, stream);
    }

    template<m_layout layout_C, m_layout layout_A, m_layout layout_B, class T, class U>
    void gemm(
        T* C, U* A, U* B, size_t M, size_t N, size_t K, size_t batch_count, hipStream_t& stream)
    {
        dispatch_fn fn = select_fn<T, U>();
        fn(C,
           A,
           B,
           M,
           N,
           K,
           batch_count,
           layout_C == m_layout::col_major ? 1 : 0,
           layout_A == m_layout::col_major ? 1 : 0,
           layout_B == m_layout::col_major ? 1 : 0,
           &stream);
    }

private:
    template<class T, class U>
    dispatch_fn select_fn() const
    {
        if constexpr(std::is_same_v<T, half> && std::is_same_v<U, half>)
        {
            return fn_f16_f16_;
        }
        else if constexpr(std::is_same_v<T, float> && std::is_same_v<U, half>)
        {
            return fn_f32_f16_;
        }
        else if constexpr(std::is_same_v<T, __hip_bfloat16> && std::is_same_v<U, __hip_bfloat16>)
        {
            return fn_bf16_bf16_;
        }
        else if constexpr(std::is_same_v<T, float> && std::is_same_v<U, __hip_bfloat16>)
        {
            return fn_f32_bf16_;
        }
        else
        {
            static_assert(sizeof(T) == 0, "Unsupported type pair for rocm_wmma_gemm::loader");
            return nullptr;
        }
    }
};

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_KERNEL_LOADER_HPP
