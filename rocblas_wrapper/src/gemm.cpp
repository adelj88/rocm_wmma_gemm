#include <gemm.hpp>
#include <hip/hip_runtime.h>

namespace rocblas_wrapper
{

bool init_rocblas()
{
    if(handle != nullptr)
    {
        return true; // Already initialized
    }

    rocblas_status status = rocblas_create_handle(&handle);
    return (status == rocblas_status_success);
}

void cleanup_rocblas()
{
    if(handle != nullptr)
    {
        rocblas_destroy_handle(handle);
        handle = nullptr;
    }
}

template<bool TRANSPOSE_A, bool TRANSPOSE_B>
__host__ void gemm(half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    if(handle == nullptr)
    {
        throw std::runtime_error("rocBLAS not initialized. Call init_rocblas() first.");
    }

    // Set stream
    rocblas_status status = rocblas_set_stream(handle, stream);
    if(status != rocblas_status_success)
    {
        throw std::runtime_error("Failed to set rocBLAS stream");
    }

    const _Float16     tmp_alpha = 1.0f;
    const _Float16     tmp_beta  = 0.0f;
    const rocblas_half alpha     = *reinterpret_cast<const rocblas_half*>(&tmp_alpha);
    const rocblas_half beta      = *reinterpret_cast<const rocblas_half*>(&tmp_beta);

    const rocblas_half* rocblas_B = reinterpret_cast<const rocblas_half*>(B);
    const rocblas_half* rocblas_A = reinterpret_cast<const rocblas_half*>(A);
    rocblas_half*       rocblas_C = reinterpret_cast<rocblas_half*>(C);

    // Perform matrix multiplication
    status = rocblas_hgemm(handle,
                           (TRANSPOSE_A) ? rocblas_operation_transpose : rocblas_operation_none,
                           (TRANSPOSE_B) ? rocblas_operation_transpose : rocblas_operation_none,
                           M, // M
                           N, // N
                           K, // K
                           &alpha,
                           rocblas_A,
                           (TRANSPOSE_A) ? K : M, // lda
                           rocblas_B,
                           (TRANSPOSE_B) ? N : K, // ldb
                           &beta,
                           rocblas_C,
                           M); // ldc

    if(status != rocblas_status_success)
    {
        throw std::runtime_error("rocBLAS HGEMM failed");
    }
}

template __host__ void
    gemm<true, true>(half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void
    gemm<true, false>(half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void
    gemm<false, true>(half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void gemm<false, false>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

} // namespace rocblas_wrapper
