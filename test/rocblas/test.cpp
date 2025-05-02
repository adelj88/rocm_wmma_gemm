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
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#include <gemm.hpp>
#include <gtest/gtest.h>
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

// Base template for kernel type wrapper
template<m_layout layout_A, m_layout layout_B>
struct LayoutWrapper
{
    static constexpr m_layout a_layout    = layout_A;
    static constexpr m_layout b_layout    = layout_B;
    static constexpr bool     a_transpose = layout_selector<layout_A>::transpose;
    static constexpr bool     b_transpose = layout_selector<layout_B>::transpose;
};

using OrderColCol = LayoutWrapper<m_layout::col_major, m_layout::col_major>;
using OrderRowCol = LayoutWrapper<m_layout::row_major, m_layout::col_major>;
using OrderColRow = LayoutWrapper<m_layout::col_major, m_layout::row_major>;
using OrderRowRow = LayoutWrapper<m_layout::row_major, m_layout::row_major>;

template<typename LayoutT>
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

    // Template function to run matrix multiplication and verify results
    void VerifyHGEMM(size_t M, size_t N, size_t K)
    {
        // Allocate memory on host using std::vector
        matrix<half, LayoutT::a_layout>   h_A(M, K);
        matrix<half, LayoutT::b_layout>   h_B(K, N);
        matrix<half, m_layout::col_major> h_C(M, N);
        matrix<half, m_layout::col_major> h_C_ref(M, N);

        // Initialize input matrices with random values
        init_matrix(h_A);
        init_matrix(h_B);

        RunTestImpl(h_A, h_B, h_C, h_C_ref, M, N, K);

        bool verification_result = verify_results(h_C, h_C_ref);
        ASSERT_TRUE(verification_result)
            << "Matrix verification failed with size " << M << "x" << N << "x" << K;
    }

private:
    // The actual test implementation in a separate method to avoid code duplication
    template<typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixCRef>
    void RunTestImpl(
        MatrixA& h_A, MatrixB& h_B, MatrixC& h_C, MatrixCRef& h_C_ref, size_t M, size_t N, size_t K)
    {
        // Allocate memory on device
        half *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, h_A.size() * sizeof(half)));
        HIP_CHECK(hipMalloc(&d_B, h_B.size() * sizeof(half)));
        HIP_CHECK(hipMalloc(&d_C, h_C.size() * sizeof(half)));

        // Copy data from host to device
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        // Execute the matrix multiplication kernel
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
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, M * N * sizeof(half), hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        // Calculate reference result on CPU
        hgemm_cpu(h_C_ref, h_A, h_B);

        // Free device memory
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
    }

    hipStream_t stream;
};

// Add or comment out type wrappers here
using LayoutTypes = ::testing::Types<OrderColCol, OrderRowCol, OrderColRow, OrderRowRow>;

TYPED_TEST_SUITE(rocBLASTest, LayoutTypes);

// Test cases for the specified matrix sizes
TYPED_TEST(rocBLASTest, Size128)
{
    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(rocBLASTest, Size128N)
{
    constexpr size_t M = 128;
    constexpr size_t N = 256;
    constexpr size_t K = 128;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(rocBLASTest, Size128K)
{
    constexpr size_t M = 128;
    constexpr size_t N = 128;
    constexpr size_t K = 256;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(rocBLASTest, Size128M)
{
    constexpr size_t M = 256;
    constexpr size_t N = 128;
    constexpr size_t K = 128;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(rocBLASTest, Size256)
{
    constexpr size_t M = 256;
    constexpr size_t N = 256;
    constexpr size_t K = 256;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(rocBLASTest, Size320)
{
    constexpr size_t M = 320;
    constexpr size_t N = 320;
    constexpr size_t K = 320;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(rocBLASTest, Size320M)
{
    constexpr size_t M = 320;
    constexpr size_t N = 512;
    constexpr size_t K = 512;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(rocBLASTest, Size320N)
{
    constexpr size_t M = 512;
    constexpr size_t N = 320;
    constexpr size_t K = 320;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(rocBLASTest, Size320K)
{
    constexpr size_t M = 512;
    constexpr size_t N = 512;
    constexpr size_t K = 320;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(rocBLASTest, Size512x320)
{
    constexpr size_t M = 512;
    constexpr size_t N = 320;
    constexpr size_t K = 512;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(rocBLASTest, Size512)
{
    constexpr size_t M = 512;
    constexpr size_t N = 512;
    constexpr size_t K = 512;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(rocBLASTest, Size1024M)
{
    constexpr size_t M = 1024;
    constexpr size_t N = 256;
    constexpr size_t K = 256;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(rocBLASTest, Size1024N)
{
    constexpr size_t M = 256;
    constexpr size_t N = 1024;
    constexpr size_t K = 256;

    std::cout << "Testing with size " << M << "x" << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
