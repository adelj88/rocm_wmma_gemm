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
#include <gtest/gtest.h>
#include <rocblas_wrapper/gemm.hpp>
#include <test.hpp>

template<m_layout layout>
struct layout_selector
{
    static constexpr bool transpose = true;
};

template<>
struct layout_selector<m_layout::col_major>
{
    static constexpr bool transpose = false;
};

template<m_layout layout_A, m_layout layout_B, m_layout layout_C>
struct LayoutWrapper
{
    static constexpr m_layout a_layout    = layout_A;
    static constexpr m_layout b_layout    = layout_B;
    static constexpr m_layout c_layout    = layout_C;
    static constexpr bool     a_transpose = layout_selector<layout_A>::transpose;
    static constexpr bool     b_transpose = layout_selector<layout_B>::transpose;
};

using OrderColColCol = LayoutWrapper<m_layout::col_major, m_layout::col_major, m_layout::col_major>;
using OrderRowColCol = LayoutWrapper<m_layout::row_major, m_layout::col_major, m_layout::col_major>;
using OrderColRowCol = LayoutWrapper<m_layout::col_major, m_layout::row_major, m_layout::col_major>;
using OrderRowRowCol = LayoutWrapper<m_layout::row_major, m_layout::row_major, m_layout::col_major>;

template<class LayoutT>
class rocBLASTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rocblas_wrapper::init_rocblas();
        HIP_CHECK(hipStreamCreate(&stream));
    }

    void TearDown() override
    {
        rocblas_wrapper::cleanup_rocblas();
        HIP_CHECK(hipStreamDestroy(stream));
    }

    // Template function to run matrix multiplication and verify results using cosine similarity
    void VerifyGEMM(size_t M, size_t N, size_t K)
    {
        // Allocate memory on host
        matrix<half, LayoutT::a_layout> h_A(M, K);
        matrix<half, LayoutT::b_layout> h_B(K, N);
        matrix<half, LayoutT::c_layout> h_C(M, N);
        matrix<half, LayoutT::c_layout> h_C_ref(M, N);

        // Initialize input matrices with random values
        init_matrix(h_A);
        init_matrix(h_B);

        // Allocate memory on device
        half *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, h_A.size() * sizeof(half)));
        HIP_CHECK(hipMalloc(&d_B, h_B.size() * sizeof(half)));
        HIP_CHECK(hipMalloc(&d_C, h_C.size() * sizeof(half)));

        // Copy data from host to device
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        // Execute rocBLAS GEMM
        rocblas_wrapper::gemm<LayoutT::a_transpose, LayoutT::b_transpose>(d_C,
                                                                          d_A,
                                                                          d_B,
                                                                          M,
                                                                          N,
                                                                          K,
                                                                          stream);
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy the result back to host
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, h_C.size() * sizeof(half), hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        // Calculate reference result on CPU
        hgemm_cpu(h_C_ref, h_A, h_B);

        // Verify results using cosine similarity
        verify_results(h_C, h_C_ref);

        // Free device memory
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
    }

private:
    hipStream_t stream;
};

using LayoutTypes
    = ::testing::Types<OrderColColCol, OrderRowColCol, OrderColRowCol, OrderRowRowCol>;

TYPED_TEST_SUITE(rocBLASTest, LayoutTypes);

TYPED_TEST(rocBLASTest, Size128x128x128)
{
    this->VerifyGEMM(128, 128, 128);
}

TYPED_TEST(rocBLASTest, Size128x256x128)
{
    this->VerifyGEMM(128, 256, 128);
}

TYPED_TEST(rocBLASTest, Size128x128x256)
{
    this->VerifyGEMM(128, 128, 256);
}

TYPED_TEST(rocBLASTest, Size256x128x128)
{
    this->VerifyGEMM(256, 128, 128);
}

TYPED_TEST(rocBLASTest, Size256x256x256)
{
    this->VerifyGEMM(256, 256, 256);
}

TYPED_TEST(rocBLASTest, Size320x320x320)
{
    this->VerifyGEMM(320, 320, 320);
}

TYPED_TEST(rocBLASTest, Size320x512x512)
{
    this->VerifyGEMM(320, 512, 512);
}

TYPED_TEST(rocBLASTest, Size512x320x320)
{
    this->VerifyGEMM(512, 320, 320);
}

TYPED_TEST(rocBLASTest, Size512x512x320)
{
    this->VerifyGEMM(512, 512, 320);
}

TYPED_TEST(rocBLASTest, Size512x320x512)
{
    this->VerifyGEMM(512, 320, 512);
}

TYPED_TEST(rocBLASTest, Size512x512x512)
{
    this->VerifyGEMM(512, 512, 512);
}

TYPED_TEST(rocBLASTest, Size1024x256x256)
{
    this->VerifyGEMM(1024, 256, 256);
}

TYPED_TEST(rocBLASTest, Size256x1024x256)
{
    this->VerifyGEMM(256, 1024, 256);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
