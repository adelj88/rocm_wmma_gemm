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

#include <common/hip_utils.hpp>
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
    constexpr float  values[] = {0.1f, 0.125f, 0.15f, 0.175f, 0.2f};
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
