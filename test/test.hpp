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

#include <common/matrix.hpp>

/**
 * @brief Initialize matrix with random values
 * @tparam T Matrix element type
 * @tparam L Matrix layout
 * @param input Matrix to initialize
 */
template<class T, m_layout L>
void init_matrix(matrix<T, L>& input)
{
    std::random_device                    rd;
    std::mt19937                          gen(1);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for(size_t m = 0; m < input.m(); ++m)
    {
        for(size_t n = 0; n < input.n(); ++n)
        {
            if constexpr(std::is_same<T, __hip_bfloat16>::value)
            {
                input(m, n) = __float2bfloat16(dis(gen));
            }
            else
            {
                input(m, n) = static_cast<T>(dis(gen));
            }
        }
    }
}

/**
 * @brief CPU reference implementation
 */
template<m_layout L1, m_layout L2, m_layout L3>
void hgemm_cpu(matrix<half, L1>& C, const matrix<half, L2>& A, const matrix<half, L3>& B)
{
    for(size_t i = 0; i < C.m(); ++i)
    {
        for(size_t j = 0; j < C.n(); ++j)
        {
            float acc = 0.0f;
            for(size_t k = 0; k < A.n(); ++k)
            {
                acc += static_cast<float>(A(i, k)) * static_cast<float>(B(k, j));
            }
            C(i, j) = static_cast<half>(acc);
        }
    }
}

template<m_layout L1, m_layout L2, m_layout L3>
void hgemm_cpu(matrix<__hip_bfloat16, L1>&       C,
               const matrix<__hip_bfloat16, L2>& A,
               const matrix<__hip_bfloat16, L3>& B)
{
    for(size_t i = 0; i < C.m(); ++i)
    {
        for(size_t j = 0; j < C.n(); ++j)
        {
            float acc = 0.0f;
            for(size_t k = 0; k < A.n(); ++k)
            {
                acc += __bfloat162float(A(i, k)) * __bfloat162float(B(k, j));
            }
            C(i, j) = __float2bfloat16(acc);
        }
    }
}

template<m_layout L1, m_layout L2, m_layout L3>
void hgemm_cpu(matrix<float, L1>& C, const matrix<half, L2>& A, const matrix<half, L3>& B)
{
    for(size_t i = 0; i < C.m(); ++i)
    {
        for(size_t j = 0; j < C.n(); ++j)
        {
            float acc = 0.0f;
            for(size_t k = 0; k < A.n(); ++k)
            {
                acc += static_cast<float>(A(i, k)) * static_cast<float>(B(k, j));
            }
            C(i, j) = acc;
        }
    }
}

template<m_layout L1, m_layout L2, m_layout L3>
void hgemm_cpu(matrix<float, L1>&                C,
               const matrix<__hip_bfloat16, L2>& A,
               const matrix<__hip_bfloat16, L3>& B)
{
    for(size_t i = 0; i < C.m(); ++i)
    {
        for(size_t j = 0; j < C.n(); ++j)
        {
            float acc = 0.0f;
            for(size_t k = 0; k < A.n(); ++k)
            {
                acc += __bfloat162float(A(i, k)) * __bfloat162float(B(k, j));
            }
            C(i, j) = acc;
        }
    }
}

/**
 * @brief Calculate cosine similarity coefficient
 */
template<class T, m_layout L>
float calculate_cosine_similarity(const matrix<T, L>& gpu_result, const matrix<T, L>& cpu_result)
{
    float dot_product = 0.0f;
    float norm_gpu    = 0.0f;
    float norm_cpu    = 0.0f;

    for(size_t i = 0; i < gpu_result.m(); ++i)
    {
        for(size_t j = 0; j < gpu_result.n(); ++j)
        {
            float gpu_val, cpu_val;
            if constexpr(std::is_same<T, __hip_bfloat16>::value)
            {
                gpu_val = __bfloat162float(gpu_result(i, j));
                cpu_val = __bfloat162float(cpu_result(i, j));
            }
            else
            {
                gpu_val = static_cast<float>(gpu_result(i, j));
                cpu_val = static_cast<float>(cpu_result(i, j));
            }

            dot_product += gpu_val * cpu_val;
            norm_gpu += gpu_val * gpu_val;
            norm_cpu += cpu_val * cpu_val;
        }
    }

    // Handle edge cases
    if(norm_gpu < 1e-10f || norm_cpu < 1e-10f)
    {
        // If either matrix is zero or near-zero, check if both are close to zero
        float max_gpu = 0.0f, max_cpu = 0.0f;
        for(size_t i = 0; i < gpu_result.m(); ++i)
        {
            for(size_t j = 0; j < gpu_result.n(); ++j)
            {
                float gpu_val, cpu_val;
                if constexpr(std::is_same<T, __hip_bfloat16>::value)
                {
                    gpu_val = __bfloat162float(gpu_result(i, j));
                    cpu_val = __bfloat162float(cpu_result(i, j));
                }
                else
                {
                    gpu_val = static_cast<float>(gpu_result(i, j));
                    cpu_val = static_cast<float>(cpu_result(i, j));
                }
                max_gpu = std::max(max_gpu, std::abs(gpu_val));
                max_cpu = std::max(max_cpu, std::abs(cpu_val));
            }
        }

        // If both matrices are near-zero, return perfect similarity
        float tolerance = std::is_same<T, __hip_bfloat16>::value ? 1e-2f : 1e-4f;
        return (max_gpu < tolerance && max_cpu < tolerance) ? 1.0f : 0.0f;
    }

    return dot_product / (std::sqrt(norm_gpu) * std::sqrt(norm_cpu));
}

/**
 * @brief Verify results using cosine similarity
 */
template<class T, m_layout L>
void verify_results(const matrix<T, L>& gpu_result, const matrix<T, L>& cpu_result)
{
    float cosine_similarity = calculate_cosine_similarity(gpu_result, cpu_result);
    float similarity_threshold;
    if constexpr(std::is_same<T, __hip_bfloat16>::value)
    {
        similarity_threshold = 0.97f; // Bfloat16 precision
    }
    else
    {
        similarity_threshold = 0.995f; // Half precision
    }

    ASSERT_GT(cosine_similarity, similarity_threshold)
        << "Cosine similarity " << cosine_similarity << " below threshold " << similarity_threshold;
}
