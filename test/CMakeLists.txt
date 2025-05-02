# Create an executable target
add_executable(gemm_test gemm/test.cpp)

# Include HIP include directories
target_include_directories(gemm_test PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(gemm_test PRIVATE rocm_wmma_gemm GTest::GTest GTest::Main)

# Create an executable target
add_executable(rocblas_test rocblas/test.cpp)

# Include HIP include directories
target_include_directories(rocblas_test PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(rocblas_test PRIVATE rocblas_wrapper GTest::GTest GTest::Main)

