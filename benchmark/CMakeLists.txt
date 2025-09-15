# Create an executable target
add_executable(bench_bf16_bf16 gemm/bench_bf16_bf16.cpp)

# Include HIP include directories
target_include_directories(bench_bf16_bf16 PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(bench_bf16_bf16 PRIVATE rocm_wmma_gemm benchmark::benchmark)

# Create an executable target
add_executable(bench_float_bf16 gemm/bench_float_bf16.cpp)

# Include HIP include directories
target_include_directories(bench_float_bf16 PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(bench_float_bf16 PRIVATE rocm_wmma_gemm benchmark::benchmark)

# Create an executable target
add_executable(bench_float_half gemm/bench_float_half.cpp)

# Include HIP include directories
target_include_directories(bench_float_half PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(bench_float_half PRIVATE rocm_wmma_gemm benchmark::benchmark)

# Create an executable target
add_executable(bench_half_half gemm/bench_half_half.cpp)

# Include HIP include directories
target_include_directories(bench_half_half PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(bench_half_half PRIVATE rocm_wmma_gemm benchmark::benchmark)

# Create an executable target
add_executable(bench_rocblas rocblas/bench.cpp)

# Include HIP include directories
target_include_directories(bench_rocblas PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(bench_rocblas PRIVATE rocblas_wrapper benchmark::benchmark)
