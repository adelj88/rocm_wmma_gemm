# Create an executable target
add_executable(gemm_bench gemm/bench.cpp)

# Include HIP include directories
target_include_directories(gemm_bench PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(gemm_bench PRIVATE rocm_wmma_gemm benchmark::benchmark)

# Create an executable target
add_executable(rocblas_bench rocblas/bench.cpp)

# Include HIP include directories
target_include_directories(rocblas_bench PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(rocblas_bench PRIVATE rocblas_wrapper benchmark::benchmark)