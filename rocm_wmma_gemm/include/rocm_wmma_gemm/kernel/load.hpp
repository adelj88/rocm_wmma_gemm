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

#ifndef ROCM_WMMA_GEMM_LOAD_HPP
#define ROCM_WMMA_GEMM_LOAD_HPP

namespace rocm_wmma_gemm
{

// Col-major load with optional padding
template<m_layout ACCESS,
         int      MAX_BITS,
         int      BLOCK_SIZE,
         int      BLOCK_M,
         int      BLOCK_N,
         int      PADDING,
         class T>
__device__ __forceinline__ auto load_to_shared(T* output, const T* input, int M, int N, int tid) ->
    typename std::enable_if<ACCESS == m_layout::col_major, void>::type
{
    // For col-major, padding is added to the row dimension (leading dimension)
    constexpr int padded_rows = BLOCK_M + PADDING;

    constexpr int min_block_dim   = (BLOCK_M < BLOCK_N) ? BLOCK_M : BLOCK_N;
    constexpr int min_block_bytes = min_block_dim * sizeof(T);

    constexpr int element_alignment = min_block_bytes / sizeof(T);
    constexpr int calculated_width  = element_alignment & (-element_alignment);
    constexpr int max_bytes         = MAX_BITS / 8;
    constexpr int max_vector_width  = max_bytes / sizeof(T);
    constexpr int actual_load_width
        = (calculated_width > max_vector_width) ? max_vector_width : calculated_width;

    using type                 = typename type_selector<T>::type;
    using vector_type          = type __attribute__((ext_vector_type(actual_load_width)));
    constexpr int vector_width = (sizeof(vector_type) / sizeof(T));

    constexpr int total_elements     = BLOCK_M * BLOCK_N;
    constexpr int total_vectors      = total_elements / vector_width;
    constexpr int vectors_per_thread = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // How many iterations are guaranteed safe for all threads?
    constexpr int guaranteed_iterations = total_vectors / BLOCK_SIZE;
    constexpr int remainder_iterations  = vectors_per_thread - guaranteed_iterations;

    const int base_idx = tid * vector_width;
    const int base_col = base_idx / BLOCK_M;
    const int base_row = base_idx % BLOCK_M;

    constexpr int col_stride    = BLOCK_SIZE * vector_width / BLOCK_M;
    constexpr int sstore_stride = col_stride * padded_rows;
    const int     gload_stride  = col_stride * M;

    // Address accumulators shared across both lambda dispatches. The unchecked
    // dispatch advances them through guaranteed_iterations steps, leaving them
    // pointing at the correct starting position for the remainder dispatch.
    // The bounds-check cursor is computed independently for the checked lambda.
    int curr_gload  = base_col * M + base_row;
    int curr_sstore = base_col * padded_rows + base_row;

    // Lambda for unchecked loads (guaranteed safe for all threads)
    auto load_vector_unchecked = [&]<size_t i>()
    {
        *reinterpret_cast<vector_type*>(output + curr_sstore)
            = *reinterpret_cast<const vector_type*>(input + curr_gload);
        curr_gload += gload_stride;
        curr_sstore += sstore_stride;
    };

    // Bounds-check cursor — computed independently from the base position and
    // the number of guaranteed iterations, rather than accumulated through
    // the unchecked lambda.
    int curr_col = base_col + guaranteed_iterations * col_stride;

    // Lambda for checked loads (only some threads have valid work)
    auto load_vector_checked = [&]<size_t i>()
    {
        if(curr_col < BLOCK_N)
        {
            *reinterpret_cast<vector_type*>(output + curr_sstore)
                = *reinterpret_cast<const vector_type*>(input + curr_gload);
        }
        curr_col += col_stride;
        curr_gload += gload_stride;
        curr_sstore += sstore_stride;
    };

    // Execute guaranteed iterations without bounds checks
    if constexpr(guaranteed_iterations > 0)
    {
        [&]<size_t... i>(std::index_sequence<i...>) {
            (load_vector_unchecked.template operator()<i>(), ...);
        }(std::make_index_sequence<guaranteed_iterations>{});
    }

    // Execute remainder iterations with bounds checks
    if constexpr(remainder_iterations > 0)
    {
        [&]<size_t... i>(std::index_sequence<i...>) {
            (load_vector_checked.template operator()<i>(), ...);
        }(std::make_index_sequence<remainder_iterations>{});
    }
}

// Row-major load with optional padding
template<m_layout ACCESS,
         int      MAX_BITS,
         int      BLOCK_SIZE,
         int      BLOCK_M,
         int      BLOCK_N,
         int      PADDING,
         class T>
__device__ __forceinline__ auto load_to_shared(T* output, const T* input, int M, int N, int tid) ->
    typename std::enable_if<ACCESS == m_layout::row_major, void>::type
{
    // For row-major, padding is added to the column dimension (leading dimension)
    constexpr int padded_cols = BLOCK_N + PADDING;

    constexpr int min_block_dim   = (BLOCK_M < BLOCK_N) ? BLOCK_M : BLOCK_N;
    constexpr int min_block_bytes = min_block_dim * sizeof(T);

    constexpr int element_alignment = min_block_bytes / sizeof(T);
    constexpr int calculated_width  = element_alignment & (-element_alignment);
    constexpr int max_bytes         = MAX_BITS / 8;
    constexpr int max_vector_width  = max_bytes / sizeof(T);
    constexpr int actual_load_width
        = (calculated_width > max_vector_width) ? max_vector_width : calculated_width;

    using type                 = typename type_selector<T>::type;
    using vector_type          = type __attribute__((ext_vector_type(actual_load_width)));
    constexpr int vector_width = (sizeof(vector_type) / sizeof(T));

    constexpr int total_elements     = BLOCK_M * BLOCK_N;
    constexpr int total_vectors      = total_elements / vector_width;
    constexpr int vectors_per_thread = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // How many iterations are guaranteed safe for all threads?
    constexpr int guaranteed_iterations = total_vectors / BLOCK_SIZE;
    constexpr int remainder_iterations  = vectors_per_thread - guaranteed_iterations;

    const int base_idx = tid * vector_width;
    const int base_row = base_idx / BLOCK_N;
    const int base_col = base_idx % BLOCK_N;

    constexpr int row_stride    = BLOCK_SIZE * vector_width / BLOCK_N;
    constexpr int sstore_stride = row_stride * padded_cols;
    const int     gload_stride  = row_stride * N;

    int curr_gload  = base_row * N + base_col;
    int curr_sstore = base_row * padded_cols + base_col;

    // Lambda for unchecked loads (guaranteed safe for all threads)
    auto load_vector_unchecked = [&]<size_t i>()
    {
        *reinterpret_cast<vector_type*>(output + curr_sstore)
            = *reinterpret_cast<const vector_type*>(input + curr_gload);
        curr_gload += gload_stride;
        curr_sstore += sstore_stride;
    };

    // Bounds-check cursor — computed independently from the base position and
    // the number of guaranteed iterations.
    int curr_row = base_row + guaranteed_iterations * row_stride;

    // Lambda for checked loads (only some threads have valid work)
    auto load_vector_checked = [&]<size_t i>()
    {
        if(curr_row < BLOCK_M)
        {
            *reinterpret_cast<vector_type*>(output + curr_sstore)
                = *reinterpret_cast<const vector_type*>(input + curr_gload);
        }
        curr_row += row_stride;
        curr_gload += gload_stride;
        curr_sstore += sstore_stride;
    };

    // Execute guaranteed iterations without bounds checks
    if constexpr(guaranteed_iterations > 0)
    {
        [&]<size_t... i>(std::index_sequence<i...>) {
            (load_vector_unchecked.template operator()<i>(), ...);
        }(std::make_index_sequence<guaranteed_iterations>{});
    }

    // Execute remainder iterations with bounds checks
    if constexpr(remainder_iterations > 0)
    {
        [&]<size_t... i>(std::index_sequence<i...>) {
            (load_vector_checked.template operator()<i>(), ...);
        }(std::make_index_sequence<remainder_iterations>{});
    }
}

template<m_layout ACCESS, int MAX_BITS, int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto
    load_shared_to_global(T* output, T* input, int row, int col, int M, int N, int tid) ->
    typename std::enable_if<ACCESS == m_layout::col_major, void>::type
{
    constexpr int min_block_dim   = (BLOCK_M < BLOCK_N) ? BLOCK_M : BLOCK_N;
    constexpr int min_block_bytes = min_block_dim * sizeof(T);

    constexpr int element_alignment = min_block_bytes / sizeof(T);
    constexpr int calculated_width  = element_alignment & (-element_alignment);
    constexpr int max_bytes         = MAX_BITS / 8;
    constexpr int max_vector_width  = max_bytes / sizeof(T);
    constexpr int actual_load_width
        = (calculated_width > max_vector_width) ? max_vector_width : calculated_width;

    using type                 = typename type_selector<T>::type;
    using vector_type          = type __attribute__((ext_vector_type(actual_load_width)));
    constexpr int vector_width = (sizeof(vector_type) / sizeof(T));

    constexpr int total_elements        = BLOCK_M * BLOCK_N;
    constexpr int total_vectors         = total_elements / vector_width;
    constexpr int vectors_per_thread    = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int guaranteed_iterations = total_vectors / BLOCK_SIZE;
    constexpr int remainder_iterations  = vectors_per_thread - guaranteed_iterations;

    const int base_idx  = tid * vector_width;
    const int base_lcol = base_idx / BLOCK_M;
    const int base_lrow = base_idx % BLOCK_M;
    const int base_grow = row + base_lrow; // invariant across iterations

    constexpr int col_stride    = BLOCK_SIZE * vector_width / BLOCK_M;
    const int     gstore_stride = col_stride * M;

    int curr_gcol   = col + base_lcol;
    int curr_gstore = curr_gcol * M + base_grow;
    int curr_sload  = base_idx;

    // Lambda for unchecked iterations (guaranteed all threads have valid shared memory indices).
    // Global boundary checks are still required as the tile may extend beyond the matrix edge.
    auto store_vector_unchecked = [&]<size_t i>()
    {
        if(curr_gcol < N && (base_grow + vector_width - 1) < M)
        {
            *reinterpret_cast<vector_type*>(output + curr_gstore)
                = *reinterpret_cast<const vector_type*>(input + curr_sload);
        }
        else if(curr_gcol < N)
        {
            for(int v = 0; v < vector_width; v++)
            {
                if((base_grow + v) < M)
                {
                    output[curr_gstore + v] = input[curr_sload + v];
                }
            }
        }
        curr_gcol += col_stride;
        curr_gstore += gstore_stride;
        curr_sload += BLOCK_SIZE * vector_width;
    };

    // Lambda for checked iterations (some threads may not have valid work)
    auto store_vector_checked = [&]<size_t i>()
    {
        if(curr_gcol < col + BLOCK_N)
        {
            if(curr_gcol < N && (base_grow + vector_width - 1) < M)
            {
                *reinterpret_cast<vector_type*>(output + curr_gstore)
                    = *reinterpret_cast<const vector_type*>(input + curr_sload);
            }
            else if(curr_gcol < N)
            {
                for(int v = 0; v < vector_width; v++)
                {
                    if((base_grow + v) < M)
                    {
                        output[curr_gstore + v] = input[curr_sload + v];
                    }
                }
            }
        }
        curr_gcol += col_stride;
        curr_gstore += gstore_stride;
        curr_sload += BLOCK_SIZE * vector_width;
    };

    // Execute guaranteed iterations
    if constexpr(guaranteed_iterations > 0)
    {
        [&]<size_t... i>(std::index_sequence<i...>) {
            (store_vector_unchecked.template operator()<i>(), ...);
        }(std::make_index_sequence<guaranteed_iterations>{});
    }

    // Execute remainder iterations
    if constexpr(remainder_iterations > 0)
    {
        [&]<size_t... i>(std::index_sequence<i...>) {
            (store_vector_checked.template operator()<i>(), ...);
        }(std::make_index_sequence<remainder_iterations>{});
    }
}

template<m_layout ACCESS, int MAX_BITS, int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto
    load_shared_to_global(T* output, T* input, int row, int col, int M, int N, int tid) ->
    typename std::enable_if<ACCESS == m_layout::row_major, void>::type
{
    constexpr int min_block_dim   = (BLOCK_M < BLOCK_N) ? BLOCK_M : BLOCK_N;
    constexpr int min_block_bytes = min_block_dim * sizeof(T);

    constexpr int element_alignment = min_block_bytes / sizeof(T);
    constexpr int calculated_width  = element_alignment & (-element_alignment);
    constexpr int max_bytes         = MAX_BITS / 8;
    constexpr int max_vector_width  = max_bytes / sizeof(T);
    constexpr int actual_load_width
        = (calculated_width > max_vector_width) ? max_vector_width : calculated_width;

    using type                 = typename type_selector<T>::type;
    using vector_type          = type __attribute__((ext_vector_type(actual_load_width)));
    constexpr int vector_width = (sizeof(vector_type) / sizeof(T));

    constexpr int total_elements        = BLOCK_M * BLOCK_N;
    constexpr int total_vectors         = total_elements / vector_width;
    constexpr int vectors_per_thread    = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int guaranteed_iterations = total_vectors / BLOCK_SIZE;
    constexpr int remainder_iterations  = vectors_per_thread - guaranteed_iterations;

    const int base_idx  = tid * vector_width;
    const int base_lrow = base_idx / BLOCK_N;
    const int base_lcol = base_idx % BLOCK_N;
    const int base_gcol = col + base_lcol; // invariant across iterations

    constexpr int row_stride    = BLOCK_SIZE * vector_width / BLOCK_N;
    const int     gstore_stride = row_stride * N;

    int curr_grow   = row + base_lrow;
    int curr_gstore = curr_grow * N + base_gcol;
    int curr_sload  = base_idx;

    // Lambda for unchecked iterations (guaranteed all threads have valid shared memory indices).
    // Global boundary checks are still required as the tile may extend beyond the matrix edge.
    auto store_vector_unchecked = [&]<size_t i>()
    {
        if(curr_grow < M && (base_gcol + vector_width - 1) < N)
        {
            *reinterpret_cast<vector_type*>(output + curr_gstore)
                = *reinterpret_cast<const vector_type*>(input + curr_sload);
        }
        else if(curr_grow < M)
        {
            for(int v = 0; v < vector_width; v++)
            {
                if((base_gcol + v) < N)
                {
                    output[curr_gstore + v] = input[curr_sload + v];
                }
            }
        }
        curr_grow += row_stride;
        curr_gstore += gstore_stride;
        curr_sload += BLOCK_SIZE * vector_width;
    };

    // Lambda for checked iterations (some threads may not have valid work)
    auto store_vector_checked = [&]<size_t i>()
    {
        if(curr_grow < row + BLOCK_M)
        {
            if(curr_grow < M && (base_gcol + vector_width - 1) < N)
            {
                *reinterpret_cast<vector_type*>(output + curr_gstore)
                    = *reinterpret_cast<const vector_type*>(input + curr_sload);
            }
            else if(curr_grow < M)
            {
                for(int v = 0; v < vector_width; v++)
                {
                    if((base_gcol + v) < N)
                    {
                        output[curr_gstore + v] = input[curr_sload + v];
                    }
                }
            }
        }
        curr_grow += row_stride;
        curr_gstore += gstore_stride;
        curr_sload += BLOCK_SIZE * vector_width;
    };

    // Execute guaranteed iterations
    if constexpr(guaranteed_iterations > 0)
    {
        [&]<size_t... i>(std::index_sequence<i...>) {
            (store_vector_unchecked.template operator()<i>(), ...);
        }(std::make_index_sequence<guaranteed_iterations>{});
    }

    // Execute remainder iterations
    if constexpr(remainder_iterations > 0)
    {
        [&]<size_t... i>(std::index_sequence<i...>) {
            (store_vector_checked.template operator()<i>(), ...);
        }(std::make_index_sequence<remainder_iterations>{});
    }
}

template<m_layout ACCESS,
         int      MAX_BITS,
         int      BLOCK_SIZE,
         int      BLOCK_M,
         int      BLOCK_N,
         int      PADDING,
         class T>
class prefetch_fragment
{
    static constexpr int min_block_dim   = (BLOCK_M < BLOCK_N) ? BLOCK_M : BLOCK_N;
    static constexpr int min_block_bytes = min_block_dim * sizeof(T);

    static constexpr int element_alignment = min_block_bytes / sizeof(T);
    static constexpr int calculated_width  = element_alignment & (-element_alignment);
    static constexpr int max_bytes         = MAX_BITS / 8;
    static constexpr int max_vector_width  = max_bytes / sizeof(T);
    static constexpr int vector_width
        = (calculated_width > max_vector_width) ? max_vector_width : calculated_width;

    static constexpr int total_elements     = BLOCK_M * BLOCK_N;
    static constexpr int total_vectors      = total_elements / vector_width;
    static constexpr int vectors_per_thread = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    static constexpr int guaranteed_iters   = total_vectors / BLOCK_SIZE;
    static constexpr int remainder_iters    = vectors_per_thread - guaranteed_iters;

    using base_type   = typename type_selector<T>::type;
    using vector_type = base_type __attribute__((ext_vector_type(vector_width)));

    vector_type regs[vectors_per_thread];

public:
    // col_major prefetch: global → registers
    template<m_layout A = ACCESS>
    __device__ __forceinline__ auto prefetch(const T* input, int M, int N, int tid) ->
        typename std::enable_if<A == m_layout::col_major, void>::type
    {
        constexpr int col_stride   = BLOCK_SIZE * vector_width / BLOCK_M;
        const int     gload_stride = col_stride * M;

        const int base_idx = tid * vector_width;
        const int base_col = base_idx / BLOCK_M;
        const int base_row = base_idx % BLOCK_M;

        int curr_gload = base_col * M + base_row;

        auto fetch_unchecked = [&]<size_t i>()
        {
            regs[i] = *reinterpret_cast<const vector_type*>(input + curr_gload);
            curr_gload += gload_stride;
        };

        // Bounds-check cursor — computed independently from the base position
        // and the number of guaranteed iterations.
        int curr_col = base_col + guaranteed_iters * col_stride;

        auto fetch_checked = [&]<size_t i>()
        {
            if(curr_col < BLOCK_N)
            {
                regs[guaranteed_iters + i]
                    = *reinterpret_cast<const vector_type*>(input + curr_gload);
            }
            curr_col += col_stride;
            curr_gload += gload_stride;
        };

        if constexpr(guaranteed_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (fetch_unchecked.template operator()<i>(), ...);
            }(std::make_index_sequence<guaranteed_iters>{});
        }

        if constexpr(remainder_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (fetch_checked.template operator()<i>(), ...);
            }(std::make_index_sequence<remainder_iters>{});
        }
    }

    // row_major prefetch: global → registers
    template<m_layout A = ACCESS>
    __device__ __forceinline__ auto prefetch(const T* input, int M, int N, int tid) ->
        typename std::enable_if<A == m_layout::row_major, void>::type
    {
        constexpr int row_stride   = BLOCK_SIZE * vector_width / BLOCK_N;
        const int     gload_stride = row_stride * N;

        const int base_idx = tid * vector_width;
        const int base_row = base_idx / BLOCK_N;
        const int base_col = base_idx % BLOCK_N;

        int curr_gload = base_row * N + base_col;

        auto fetch_unchecked = [&]<size_t i>()
        {
            regs[i] = *reinterpret_cast<const vector_type*>(input + curr_gload);
            curr_gload += gload_stride;
        };

        // Bounds-check cursor — computed independently from the base position
        // and the number of guaranteed iterations.
        int curr_row = base_row + guaranteed_iters * row_stride;

        auto fetch_checked = [&]<size_t i>()
        {
            if(curr_row < BLOCK_M)
            {
                regs[guaranteed_iters + i]
                    = *reinterpret_cast<const vector_type*>(input + curr_gload);
            }
            curr_row += row_stride;
            curr_gload += gload_stride;
        };

        if constexpr(guaranteed_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (fetch_unchecked.template operator()<i>(), ...);
            }(std::make_index_sequence<guaranteed_iters>{});
        }

        if constexpr(remainder_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (fetch_checked.template operator()<i>(), ...);
            }(std::make_index_sequence<remainder_iters>{});
        }
    }

    // col_major partial_prefetch: global → registers (slice STEP of TOTAL_STEPS)
    template<size_t STEP, size_t TOTAL_STEPS, m_layout A = ACCESS>
    __device__ __forceinline__ auto partial_prefetch(const T* input, int M, int N, int tid) ->
        typename std::enable_if<A == m_layout::col_major, void>::type
    {
        constexpr size_t vpt            = static_cast<size_t>(vectors_per_thread);
        constexpr size_t loads_per_step = (vpt + TOTAL_STEPS - 1) / TOTAL_STEPS;
        constexpr size_t start          = STEP * loads_per_step;
        constexpr size_t end = (start + loads_per_step > vpt) ? vpt : start + loads_per_step;

        constexpr size_t guaranteed_end   = (end < guaranteed_iters) ? end : guaranteed_iters;
        constexpr size_t guaranteed_count = (start < guaranteed_end) ? guaranteed_end - start : 0;

        constexpr size_t remainder_start = (start < guaranteed_iters) ? guaranteed_iters : start;
        constexpr size_t remainder_end   = end;
        constexpr size_t remainder_count
            = (remainder_start < remainder_end) ? remainder_end - remainder_start : 0;

        if constexpr(guaranteed_count == 0 && remainder_count == 0)
        {
            return;
        }

        constexpr int col_stride   = BLOCK_SIZE * vector_width / BLOCK_M;
        const int     gload_stride = col_stride * M;

        const int base_idx = tid * vector_width;
        const int base_col = base_idx / BLOCK_M;
        const int base_row = base_idx % BLOCK_M;

        // Address accumulator shared across both lambda dispatches. The
        // unchecked dispatch advances it through the guaranteed portion of
        // this step, leaving it pointing at the correct starting position
        // for the remainder dispatch.
        int curr_gload = (base_col * M + base_row) + static_cast<int>(start) * gload_stride;

        // Lambda for unchecked loads (guaranteed safe for all threads)
        auto fetch_unchecked = [&]<size_t i>()
        {
            regs[static_cast<size_t>(start) + i]
                = *reinterpret_cast<const vector_type*>(input + curr_gload);
            curr_gload += gload_stride;
        };

        // Bounds-check cursor — computed independently from the base position
        // and the number of guaranteed iterations in this step.
        int curr_col = base_col + static_cast<int>(start + guaranteed_count) * col_stride;

        // Lambda for checked loads (only some threads have valid work)
        auto fetch_checked = [&]<size_t i>()
        {
            if(curr_col < BLOCK_N)
            {
                regs[remainder_start + i]
                    = *reinterpret_cast<const vector_type*>(input + curr_gload);
            }
            curr_col += col_stride;
            curr_gload += gload_stride;
        };

        // Execute guaranteed portion of this step without bounds checks
        if constexpr(guaranteed_count > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (fetch_unchecked.template operator()<i>(), ...);
            }(std::make_index_sequence<guaranteed_count>{});
        }

        // Execute remainder portion of this step with bounds checks
        if constexpr(remainder_count > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (fetch_checked.template operator()<i>(), ...);
            }(std::make_index_sequence<remainder_count>{});
        }
    }

    // row_major partial_prefetch: global → registers (slice STEP of TOTAL_STEPS)
    template<size_t STEP, size_t TOTAL_STEPS, m_layout A = ACCESS>
    __device__ __forceinline__ auto partial_prefetch(const T* input, int M, int N, int tid) ->
        typename std::enable_if<A == m_layout::row_major, void>::type
    {
        constexpr size_t vpt            = static_cast<size_t>(vectors_per_thread);
        constexpr size_t loads_per_step = (vpt + TOTAL_STEPS - 1) / TOTAL_STEPS;
        constexpr size_t start          = STEP * loads_per_step;
        constexpr size_t end = (start + loads_per_step > vpt) ? vpt : start + loads_per_step;

        constexpr size_t guaranteed_end   = (end < guaranteed_iters) ? end : guaranteed_iters;
        constexpr size_t guaranteed_count = (start < guaranteed_end) ? guaranteed_end - start : 0;

        constexpr size_t remainder_start = (start < guaranteed_iters) ? guaranteed_iters : start;
        constexpr size_t remainder_end   = end;
        constexpr size_t remainder_count
            = (remainder_start < remainder_end) ? remainder_end - remainder_start : 0;

        if constexpr(guaranteed_count == 0 && remainder_count == 0)
        {
            return;
        }

        constexpr int row_stride   = BLOCK_SIZE * vector_width / BLOCK_N;
        const int     gload_stride = row_stride * N;

        const int base_idx = tid * vector_width;
        const int base_row = base_idx / BLOCK_N;
        const int base_col = base_idx % BLOCK_N;

        // Address accumulator shared across both lambda dispatches. The
        // unchecked dispatch advances it through the guaranteed portion of
        // this step, leaving it pointing at the correct starting position
        // for the remainder dispatch.
        int curr_gload = (base_row * N + base_col) + static_cast<int>(start) * gload_stride;

        // Lambda for unchecked loads (guaranteed safe for all threads)
        auto fetch_unchecked = [&]<size_t i>()
        {
            regs[static_cast<size_t>(start) + i]
                = *reinterpret_cast<const vector_type*>(input + curr_gload);
            curr_gload += gload_stride;
        };

        // Bounds-check cursor — computed independently from the base position
        // and the number of guaranteed iterations in this step.
        int curr_row = base_row + static_cast<int>(start + guaranteed_count) * row_stride;

        // Lambda for checked loads (only some threads have valid work)
        auto fetch_checked = [&]<size_t i>()
        {
            if(curr_row < BLOCK_M)
            {
                regs[remainder_start + i]
                    = *reinterpret_cast<const vector_type*>(input + curr_gload);
            }
            curr_row += row_stride;
            curr_gload += gload_stride;
        };

        // Execute guaranteed portion of this step without bounds checks
        if constexpr(guaranteed_count > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (fetch_unchecked.template operator()<i>(), ...);
            }(std::make_index_sequence<guaranteed_count>{});
        }

        // Execute remainder portion of this step with bounds checks
        if constexpr(remainder_count > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (fetch_checked.template operator()<i>(), ...);
            }(std::make_index_sequence<remainder_count>{});
        }
    }

    // col_major partial_commit: registers → LDS (slice STEP of TOTAL_STEPS)
    template<size_t STEP, size_t TOTAL_STEPS, m_layout A = ACCESS>
    __device__ __forceinline__ auto partial_commit(T* output, int tid) ->
        typename std::enable_if<A == m_layout::col_major, void>::type
    {
        constexpr size_t vpt            = static_cast<size_t>(vectors_per_thread);
        constexpr size_t loads_per_step = (vpt + TOTAL_STEPS - 1) / TOTAL_STEPS;
        constexpr size_t start          = STEP * loads_per_step;
        constexpr size_t end = (start + loads_per_step > vpt) ? vpt : start + loads_per_step;

        constexpr size_t guaranteed_end   = (end < guaranteed_iters) ? end : guaranteed_iters;
        constexpr size_t guaranteed_count = (start < guaranteed_end) ? guaranteed_end - start : 0;

        constexpr size_t remainder_start = (start < guaranteed_iters) ? guaranteed_iters : start;
        constexpr size_t remainder_end   = end;
        constexpr size_t remainder_count
            = (remainder_start < remainder_end) ? remainder_end - remainder_start : 0;

        if constexpr(guaranteed_count == 0 && remainder_count == 0)
        {
            return;
        }

        constexpr int padded_rows   = BLOCK_M + PADDING;
        constexpr int col_stride    = BLOCK_SIZE * vector_width / BLOCK_M;
        constexpr int sstore_stride = col_stride * padded_rows;

        const int base_idx = tid * vector_width;
        const int base_col = base_idx / BLOCK_M;
        const int base_row = base_idx % BLOCK_M;

        // Address accumulator shared across both lambda dispatches. The
        // unchecked dispatch advances it through the guaranteed portion of
        // this step, leaving it pointing at the correct starting position
        // for the remainder dispatch.
        int curr_sstore
            = (base_col * padded_rows + base_row) + static_cast<int>(start) * sstore_stride;

        // Lambda for unchecked stores (guaranteed safe for all threads)
        auto store_unchecked = [&]<size_t i>()
        {
            *reinterpret_cast<vector_type*>(output + curr_sstore)
                = regs[static_cast<size_t>(start) + i];
            curr_sstore += sstore_stride;
        };

        // Bounds-check cursor — computed independently from the base position
        // and the number of guaranteed iterations in this step.
        int curr_col = base_col + static_cast<int>(start + guaranteed_count) * col_stride;

        // Lambda for checked stores (only some threads have valid work)
        auto store_checked = [&]<size_t i>()
        {
            if(curr_col < BLOCK_N)
            {
                *reinterpret_cast<vector_type*>(output + curr_sstore) = regs[remainder_start + i];
            }
            curr_col += col_stride;
            curr_sstore += sstore_stride;
        };

        // Execute guaranteed portion of this step without bounds checks
        if constexpr(guaranteed_count > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_unchecked.template operator()<i>(), ...);
            }(std::make_index_sequence<guaranteed_count>{});
        }

        // Execute remainder portion of this step with bounds checks
        if constexpr(remainder_count > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_checked.template operator()<i>(), ...);
            }(std::make_index_sequence<remainder_count>{});
        }
    }

    // row_major partial_commit: registers → LDS (slice STEP of TOTAL_STEPS)
    template<size_t STEP, size_t TOTAL_STEPS, m_layout A = ACCESS>
    __device__ __forceinline__ auto partial_commit(T* output, int tid) ->
        typename std::enable_if<A == m_layout::row_major, void>::type
    {
        constexpr size_t vpt            = static_cast<size_t>(vectors_per_thread);
        constexpr size_t loads_per_step = (vpt + TOTAL_STEPS - 1) / TOTAL_STEPS;
        constexpr size_t start          = STEP * loads_per_step;
        constexpr size_t end = (start + loads_per_step > vpt) ? vpt : start + loads_per_step;

        constexpr size_t guaranteed_end   = (end < guaranteed_iters) ? end : guaranteed_iters;
        constexpr size_t guaranteed_count = (start < guaranteed_end) ? guaranteed_end - start : 0;

        constexpr size_t remainder_start = (start < guaranteed_iters) ? guaranteed_iters : start;
        constexpr size_t remainder_end   = end;
        constexpr size_t remainder_count
            = (remainder_start < remainder_end) ? remainder_end - remainder_start : 0;

        if constexpr(guaranteed_count == 0 && remainder_count == 0)
        {
            return;
        }

        constexpr int padded_cols   = BLOCK_N + PADDING;
        constexpr int row_stride    = BLOCK_SIZE * vector_width / BLOCK_N;
        constexpr int sstore_stride = row_stride * padded_cols;

        const int base_idx = tid * vector_width;
        const int base_row = base_idx / BLOCK_N;
        const int base_col = base_idx % BLOCK_N;

        // Address accumulator shared across both lambda dispatches. The
        // unchecked dispatch advances it through the guaranteed portion of
        // this step, leaving it pointing at the correct starting position
        // for the remainder dispatch.
        int curr_sstore
            = (base_row * padded_cols + base_col) + static_cast<int>(start) * sstore_stride;

        // Lambda for unchecked stores (guaranteed safe for all threads)
        auto store_unchecked = [&]<size_t i>()
        {
            *reinterpret_cast<vector_type*>(output + curr_sstore)
                = regs[static_cast<size_t>(start) + i];
            curr_sstore += sstore_stride;
        };

        // Bounds-check cursor — computed independently from the base position
        // and the number of guaranteed iterations in this step.
        int curr_row = base_row + static_cast<int>(start + guaranteed_count) * row_stride;

        // Lambda for checked stores (only some threads have valid work)
        auto store_checked = [&]<size_t i>()
        {
            if(curr_row < BLOCK_M)
            {
                *reinterpret_cast<vector_type*>(output + curr_sstore) = regs[remainder_start + i];
            }
            curr_row += row_stride;
            curr_sstore += sstore_stride;
        };

        // Execute guaranteed portion of this step without bounds checks
        if constexpr(guaranteed_count > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_unchecked.template operator()<i>(), ...);
            }(std::make_index_sequence<guaranteed_count>{});
        }

        // Execute remainder portion of this step with bounds checks
        if constexpr(remainder_count > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_checked.template operator()<i>(), ...);
            }(std::make_index_sequence<remainder_count>{});
        }
    }

    // col_major commit: registers → LDS (caller must ensure vmcnt(0) before calling)
    template<m_layout A = ACCESS>
    __device__ __forceinline__ auto commit(T* output, int tid) ->
        typename std::enable_if<A == m_layout::col_major, void>::type
    {
        constexpr int padded_rows   = BLOCK_M + PADDING;
        constexpr int col_stride    = BLOCK_SIZE * vector_width / BLOCK_M;
        constexpr int sstore_stride = col_stride * padded_rows;

        const int base_idx = tid * vector_width;
        const int base_col = base_idx / BLOCK_M;
        const int base_row = base_idx % BLOCK_M;

        int curr_sstore = base_col * padded_rows + base_row;

        auto store_unchecked = [&]<size_t i>()
        {
            *reinterpret_cast<vector_type*>(output + curr_sstore) = regs[i];
            curr_sstore += sstore_stride;
        };

        // Bounds-check cursor — computed independently from the base position
        // and the number of guaranteed iterations.
        int curr_col = base_col + guaranteed_iters * col_stride;

        auto store_checked = [&]<size_t i>()
        {
            if(curr_col < BLOCK_N)
            {
                *reinterpret_cast<vector_type*>(output + curr_sstore) = regs[guaranteed_iters + i];
            }
            curr_col += col_stride;
            curr_sstore += sstore_stride;
        };

        if constexpr(guaranteed_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_unchecked.template operator()<i>(), ...);
            }(std::make_index_sequence<guaranteed_iters>{});
        }

        if constexpr(remainder_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_checked.template operator()<i>(), ...);
            }(std::make_index_sequence<remainder_iters>{});
        }
    }

    // row_major commit: registers → LDS (caller must ensure vmcnt(0) before calling)
    template<m_layout A = ACCESS>
    __device__ __forceinline__ auto commit(T* output, int tid) ->
        typename std::enable_if<A == m_layout::row_major, void>::type
    {
        constexpr int padded_cols   = BLOCK_N + PADDING;
        constexpr int row_stride    = BLOCK_SIZE * vector_width / BLOCK_N;
        constexpr int sstore_stride = row_stride * padded_cols;

        const int base_idx = tid * vector_width;
        const int base_row = base_idx / BLOCK_N;
        const int base_col = base_idx % BLOCK_N;

        int curr_sstore = base_row * padded_cols + base_col;

        auto store_unchecked = [&]<size_t i>()
        {
            *reinterpret_cast<vector_type*>(output + curr_sstore) = regs[i];
            curr_sstore += sstore_stride;
        };

        // Bounds-check cursor — computed independently from the base position
        // and the number of guaranteed iterations.
        int curr_row = base_row + guaranteed_iters * row_stride;

        auto store_checked = [&]<size_t i>()
        {
            if(curr_row < BLOCK_M)
            {
                *reinterpret_cast<vector_type*>(output + curr_sstore) = regs[guaranteed_iters + i];
            }
            curr_row += row_stride;
            curr_sstore += sstore_stride;
        };

        if constexpr(guaranteed_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_unchecked.template operator()<i>(), ...);
            }(std::make_index_sequence<guaranteed_iters>{});
        }

        if constexpr(remainder_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_checked.template operator()<i>(), ...);
            }(std::make_index_sequence<remainder_iters>{});
        }
    }
};

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_LOAD_HPP
