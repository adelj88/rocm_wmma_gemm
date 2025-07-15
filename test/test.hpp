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
template<m_layout L1, m_layout L2, m_layout L3, class T>
void hgemm_cpu(matrix<T, L1>& C, const matrix<T, L2>& A, const matrix<T, L3>& B)
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
            C(i, j) = static_cast<T>(acc);
        }
    }
}

template<m_layout L1, m_layout L2, m_layout L3, class T, class U>
void hgemm_cpu(matrix<T, L1>& C, const matrix<U, L2>& A, const matrix<U, L3>& B)
{
    for(size_t i = 0; i < C.m(); ++i)
    {
        for(size_t j = 0; j < C.n(); ++j)
        {
            T acc = T(0);
            for(size_t k = 0; k < A.n(); ++k)
            {
                acc += static_cast<T>(A(i, k)) * static_cast<T>(B(k, j));
            }
            C(i, j) = acc;
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
    // Tolerance checking parameters
    float atol                   = 0.14f; // absolute tolerance
    float rtol                   = 0.0f; // relative tolerance
    bool  tolerance_check_passed = true;
    float max_violation          = 0.0f;
#endif

    // Single pass: calculate means, max errors, and collect data for SSIM
    // Also find min/max for dynamic range calculation
    float sum_gpu = 0.0f, sum_cpu = 0.0f;
    float min_gpu = std::numeric_limits<float>::max();
    float max_gpu = std::numeric_limits<float>::lowest();
    float min_cpu = std::numeric_limits<float>::max();
    float max_cpu = std::numeric_limits<float>::lowest();

    for(size_t i = 0; i < m; ++i)
    {
        for(size_t j = 0; j < n; ++j)
        {
            float gpu_val = static_cast<float>(gpu_result(i, j));
            float cpu_val = static_cast<float>(cpu_result(i, j));

            sum_gpu += gpu_val;
            sum_cpu += cpu_val;

            // Track min/max for dynamic range
            min_gpu = std::min(min_gpu, gpu_val);
            max_gpu = std::max(max_gpu, gpu_val);
            min_cpu = std::min(min_cpu, cpu_val);
            max_cpu = std::max(max_cpu, cpu_val);

#ifdef ELEMENT_CHECK
            // Element-wise tolerance checking
            float abs_diff  = std::abs(gpu_val - cpu_val);
            float tolerance = atol + rtol * std::abs(cpu_val);

            if(abs_diff > tolerance && tolerance_check_passed)
            {
                tolerance_check_passed = false;
                max_violation          = abs_diff - tolerance;
                metrics.error_i        = i;
                metrics.error_j        = j;
                metrics.gpu_val        = gpu_val;
                metrics.cpu_val        = cpu_val;
            }
#endif
        }
    }

    float mean_gpu = sum_gpu / total_elements;
    float mean_cpu = sum_cpu / total_elements;

    // Calculate dynamic range for SSIM constants
    float range_gpu     = max_gpu - min_gpu;
    float range_cpu     = max_cpu - min_cpu;
    float dynamic_range = std::max(range_gpu, range_cpu);

    // Ensure minimum dynamic range to avoid division issues
    if(dynamic_range < 1e-6f)
    {
        dynamic_range = 1.0f;
    }

    // Second pass: calculate SSIM components using raw values (not absolute)
    float var_gpu = 0.0f, var_cpu = 0.0f, covar = 0.0f;
    for(size_t i = 0; i < m; ++i)
    {
        for(size_t j = 0; j < n; ++j)
        {
            float gpu_val = static_cast<float>(gpu_result(i, j));
            float cpu_val = static_cast<float>(cpu_result(i, j));

            float gpu_diff = gpu_val - mean_gpu;
            float cpu_diff = cpu_val - mean_cpu;

            var_gpu += gpu_diff * gpu_diff;
            var_cpu += cpu_diff * cpu_diff;
            covar += gpu_diff * cpu_diff;
        }
    }

    var_gpu /= total_elements;
    var_cpu /= total_elements;
    covar /= total_elements;

    // Standard SSIM constants based on dynamic range
    const float C1 = (0.01f * dynamic_range) * (0.01f * dynamic_range);
    const float C2 = (0.03f * dynamic_range) * (0.03f * dynamic_range);

    metrics.ssim = ((2 * mean_gpu * mean_cpu + C1) * (2 * covar + C2))
                   / ((mean_gpu * mean_gpu + mean_cpu * mean_cpu + C1) * (var_gpu + var_cpu + C2));

#ifdef ELEMENT_CHECK
    // Store tolerance checking results
    metrics.max_relative_error = tolerance_check_passed ? 0.0f : max_violation;
    metrics.tolerance          = tolerance_check_passed ? 1.0f : 0.0f; // Use as boolean flag
#endif

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
    // Element-wise tolerance validation
    ASSERT_EQ(metrics.tolerance, 1.0f)
        << "Element-wise tolerance check failed. Max violation: " << metrics.max_relative_error
        << " at position (" << metrics.error_i << "," << metrics.error_j << ")"
        << " GPU=" << metrics.gpu_val << " CPU=" << metrics.cpu_val;
#endif

    // SSIM structural validation
    ASSERT_GT(metrics.ssim, 0.98f) << "SSIM " << metrics.ssim << " below threshold 0.98";
}
