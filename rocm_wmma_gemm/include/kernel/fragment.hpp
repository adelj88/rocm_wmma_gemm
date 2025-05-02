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

#ifndef ROCM_WMMA_GEMM_FRAGMENT_HPP
#define ROCM_WMMA_GEMM_FRAGMENT_HPP

namespace rocm_wmma_gemm
{

template<class T>
struct type_selector
{
    using type = T;
};

template<>
struct type_selector<half>
{
    using type = _Float16;
};

template<class T, int TILE>
class fragment
{
public:
    using underlying_type = T;
    using type            = typename type_selector<T>::type;
    typedef type frag_vec __attribute__((ext_vector_type(TILE)));
    using value_type = type;

private:
    frag_vec _fragment = {};

public:
    class proxy
    {
        frag_vec& vec_ref;
        int       index;

        friend class iterator;

    public:
        __device__ proxy(frag_vec& v, int i) : vec_ref(v), index(i) {}

        __device__ proxy& operator=(type value)
        {
            vec_ref[index] = value;
            return *this;
        }

        __device__ proxy& operator=(const T& value)
        {
            vec_ref[index] = static_cast<type>(value);
            return *this;
        }

        __device__ operator type() const
        {
            return vec_ref[index];
        }

        proxy*       operator&()       = delete;
        const proxy* operator&() const = delete;
        proxy(const proxy&)            = delete;
    };

    class iterator
    {
        frag_vec& vec_ref;
        int       current_index;

        friend class fragment<T, TILE>;

        __device__ iterator(frag_vec& v, int i) : vec_ref(v), current_index(i) {}

    public:
        __device__ proxy operator*() const
        {
            return proxy(vec_ref, current_index);
        }

        __device__ iterator& operator++()
        {
            ++current_index;
            return *this;
        }

        __device__ iterator& operator+=(int n)
        {
            current_index += n;
            return *this;
        }

        __device__ iterator operator+(int n) const
        {
            iterator temp = *this;
            temp += n;
            return temp;
        }

        __device__ bool operator!=(const iterator& other) const
        {
            return current_index != other.current_index;
        }
    };

public:
    __device__ iterator begin()
    {
        return iterator(_fragment, 0);
    }

    __device__ iterator end()
    {
        return iterator(_fragment, TILE);
    }

    __device__ frag_vec& get()
    {
        return _fragment;
    }

    __device__ const frag_vec& get() const
    {
        return _fragment;
    }

    __device__ type operator[](int i) const
    {
        return _fragment[i];
    }
};

template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_a && ACCESS == m_layout::row_major),
                            void>::type
{
    using type = typename type_selector<T>::type;
    typedef type vec __attribute__((ext_vector_type(TILE)));
    const vec*   tmp = reinterpret_cast<const vec*>(data);
    frag.get()       = *tmp;
}

template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_a && ACCESS == m_layout::col_major),
                            void>::type
{
    const T* tmp = reinterpret_cast<const T*>(data);
    for(auto it = frag.begin(); it != frag.end(); ++it)
    {
        *it = *tmp;
        tmp += M;
    }
}

template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_b && ACCESS == m_layout::row_major),
                            void>::type
{
    const T* tmp = reinterpret_cast<const T*>(data);
    for(auto it = frag.begin(); it != frag.end(); ++it)
    {
        *it = *tmp;
        tmp += N;
    }
}

template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_b && ACCESS == m_layout::col_major),
                            void>::type
{
    using type = typename type_selector<T>::type;
    typedef type vec __attribute__((ext_vector_type(TILE)));
    const vec*   tmp = reinterpret_cast<const vec*>(data);
    frag.get()       = *tmp;
}

template<m_layout ACCESS, bool USE_SHARED, class T, int TILE>
__device__ __forceinline__ auto
    store_matrix(T* data, fragment<T, TILE>& frag, int row, int col, int M, int N) ->
    typename std::enable_if<ACCESS == m_layout::row_major && !USE_SHARED, void>::type
{
#pragma unroll
    for(int i = 0; i < TILE / 2; ++i)
    {
        const int r = i * 2;
        if((row + r) < M && col < N)
        {
            data[(row + r) * N + col] = frag[r];
        }
    }
}

template<m_layout ACCESS, bool USE_SHARED, class T, int TILE>
__device__ __forceinline__ auto
    store_matrix(T* data, fragment<T, TILE>& frag, int row, int col, int M, int N) ->
    typename std::enable_if<ACCESS == m_layout::row_major && USE_SHARED, void>::type
{
#pragma unroll
    for(int i = 0; i < TILE / 2; ++i)
    {
        const int r               = i * 2;
        data[(row + r) * N + col] = frag[r];
    }
}

template<m_layout ACCESS, bool USE_SHARED, class T, int TILE>
__device__ __forceinline__ auto
    store_matrix(T* data, fragment<T, TILE>& frag, int row, int col, int M, int N) ->
    typename std::enable_if<ACCESS == m_layout::col_major && !USE_SHARED, void>::type
{
#pragma unroll
    for(int i = 0; i < TILE / 2; ++i)
    {
        const int r = i * 2;
        if((row + r) < M && col < N)
        {
            data[col * M + (row + r)] = frag[r];
        }
    }
}

template<m_layout ACCESS, bool USE_SHARED, class T, int TILE>
__device__ __forceinline__ auto
    store_matrix(T* data, fragment<T, TILE>& frag, int row, int col, int M, int N) ->
    typename std::enable_if<ACCESS == m_layout::col_major && USE_SHARED, void>::type
{
#pragma unroll
    for(int i = 0; i < TILE / 2; ++i)
    {
        const int r               = i * 2;
        data[col * M + (row + r)] = frag[r];
    }
}

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_FRAGMENT_HPP
