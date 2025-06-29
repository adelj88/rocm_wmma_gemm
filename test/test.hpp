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
    constexpr float  values[]   = {0.1f, 0.125f, 0.15f, 0.175f, 0.2f};
    constexpr size_t num_values = sizeof(values) / sizeof(values[0]);

    size_t idx = 0;
    for(size_t m = 0; m < input.m(); ++m)
    {
        for(size_t n = 0; n < input.n(); ++n)
        {
            input(m, n) = static_cast<T>(values[idx % num_values]);
            idx++;
        }
    }
}

/**
 * @brief Initialize matrix as identity matrix
 */
template<typename T, m_layout L>
void init_identity_matrix(matrix<T, L>& identity)
{
    for(size_t i = 0; i < identity.m(); ++i)
    {
        for(size_t j = 0; j < identity.n(); ++j)
        {
            identity(i, j) = (i == j) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
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

/**
 * @brief Verification metrics structure
 */
struct verification_metrics
{
    float  max_relative_error;
    float  avg_relative_error;
    float  ssim;
    float  tolerance;
    size_t error_i, error_j;
    float  gpu_val, cpu_val;
    size_t valid_comparisons;
};

/**
 * @brief Calculate verification metrics
 */
template<m_layout L>
verification_metrics calculate_metrics(const matrix<half, L>& gpu_result,
                                       const matrix<half, L>& cpu_result)
{
    verification_metrics metrics = {};

    size_t m              = gpu_result.m();
    size_t n              = gpu_result.n();
    size_t total_elements = m * n;

#ifdef ELEMENT_CHECK
    // Scale tolerance based on matrix size
    float size_factor = std::log2(std::max(m, n)) / 8.0f;
    metrics.tolerance = 0.02f + 0.02f * size_factor;
#endif

    // Single pass: calculate means, max errors, and collect data for SSIM
    float sum_gpu = 0.0f, sum_cpu = 0.0f;
    float sum_rel_diff = 0.0f;
    float sum_gpu_abs = 0.0f, sum_cpu_abs = 0.0f; // For SSIM calculation

    for(size_t i = 0; i < m; ++i)
    {
        for(size_t j = 0; j < n; ++j)
        {
            float gpu_val = static_cast<float>(gpu_result(i, j));
            float cpu_val = static_cast<float>(cpu_result(i, j));

            sum_gpu += gpu_val;
            sum_cpu += cpu_val;

            // Use absolute values for SSIM
            sum_gpu_abs += std::abs(gpu_val);
            sum_cpu_abs += std::abs(cpu_val);

#ifdef ELEMENT_CHECK
            float rel_diff = std::abs(gpu_val - cpu_val) / std::abs(cpu_val);
            sum_rel_diff += rel_diff;
            metrics.valid_comparisons++;

            if(rel_diff > metrics.max_relative_error)
            {
                metrics.max_relative_error = rel_diff;
                metrics.error_i            = i;
                metrics.error_j            = j;
                metrics.gpu_val            = gpu_val;
                metrics.cpu_val            = cpu_val;
            }
#endif
        }
    }

    float mean_gpu = sum_gpu / total_elements;
    float mean_cpu = sum_cpu / total_elements;
#ifdef ELEMENT_CHECK
    metrics.avg_relative_error
        = metrics.valid_comparisons > 0 ? sum_rel_diff / metrics.valid_comparisons : 0.0f;
#endif

    // SSIM calculation using absolute values
    float mean_gpu_abs = sum_gpu_abs / total_elements;
    float mean_cpu_abs = sum_cpu_abs / total_elements;

    // Second pass: calculate SSIM components using absolute values
    float var_gpu_abs = 0.0f, var_cpu_abs = 0.0f, covar_abs = 0.0f;
    for(size_t i = 0; i < m; ++i)
    {
        for(size_t j = 0; j < n; ++j)
        {
            float gpu_val_abs = std::abs(static_cast<float>(gpu_result(i, j)));
            float cpu_val_abs = std::abs(static_cast<float>(cpu_result(i, j)));

            float gpu_diff_abs = gpu_val_abs - mean_gpu_abs;
            float cpu_diff_abs = cpu_val_abs - mean_cpu_abs;

            var_gpu_abs += gpu_diff_abs * gpu_diff_abs;
            var_cpu_abs += cpu_diff_abs * cpu_diff_abs;
            covar_abs += gpu_diff_abs * cpu_diff_abs;
        }
    }

    var_gpu_abs /= total_elements;
    var_cpu_abs /= total_elements;
    covar_abs /= total_elements;

    // Original SSIM formula, but with absolute values (always positive)
    const float C1 = 0.01f * mean_cpu_abs * mean_cpu_abs;
    const float C2 = 0.03f * var_cpu_abs;

    metrics.ssim = ((2 * mean_gpu_abs * mean_cpu_abs + C1) * (2 * covar_abs + C2))
                   / ((mean_gpu_abs * mean_gpu_abs + mean_cpu_abs * mean_cpu_abs + C1)
                      * (var_gpu_abs + var_cpu_abs + C2));

    return metrics;
}

/**
 * @brief Verify results with separate assertions
 */
template<m_layout L>
void verify_results(const matrix<half, L>& gpu_result, const matrix<half, L>& cpu_result)
{
    verification_metrics metrics = calculate_metrics(gpu_result, cpu_result);

#ifdef ELEMENT_CHECK
    // Element-wise validation
    ASSERT_LE(metrics.max_relative_error, metrics.tolerance)
        << "Maximum relative error " << metrics.max_relative_error << " exceeds tolerance "
        << metrics.tolerance << " at position (" << metrics.error_i << "," << metrics.error_j << ")"
        << " GPU=" << metrics.gpu_val << " CPU=" << metrics.cpu_val;
#endif

    // Pattern validation
    ASSERT_GT(metrics.ssim, 0.98f) << "SSIM " << metrics.ssim << " below threshold 0.98"
                                   << " (avg error: " << metrics.avg_relative_error << ")";
}
