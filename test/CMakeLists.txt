# Create an executable target
add_executable(test_float_accum gemm/test_float_accum.cpp)

# Include HIP include directories
target_include_directories(test_float_accum PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(test_float_accum PRIVATE rocm_wmma_gemm GTest::GTest GTest::Main)

# Create an executable target
add_executable(test_same_prec gemm/test_same_prec.cpp)

# Include HIP include directories
target_include_directories(test_same_prec PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(test_same_prec PRIVATE rocm_wmma_gemm GTest::GTest GTest::Main)

# Create an executable target
add_executable(rocblas_test rocblas/test.cpp)

# Include HIP include directories
target_include_directories(rocblas_test PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(rocblas_test PRIVATE rocblas_wrapper GTest::GTest GTest::Main)

