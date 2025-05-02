set_source_files_properties(src/gemm.cpp PROPERTIES COMPILE_OPTIONS -mcumode)

add_library(rocm_wmma_gemm STATIC src/gemm.cpp)

# Include HIP include directories
target_include_directories(rocm_wmma_gemm PUBLIC ${HIP_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link HIP libraries
target_link_libraries(rocm_wmma_gemm PUBLIC ${HIP_LIBRARIES})

# Add optimization flags
target_compile_options(rocm_wmma_gemm PRIVATE
    -O3                 # High optimization
    -ffast-math         # Fast math operations (may break strict IEEE compliance)
)