add_library(rocblas_wrapper STATIC src/gemm.cpp)

# Include HIP include directories
target_include_directories(rocblas_wrapper PUBLIC ${HIP_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link HIP libraries
target_link_libraries(rocblas_wrapper PUBLIC roc::rocblas ${HIP_LIBRARIES})
