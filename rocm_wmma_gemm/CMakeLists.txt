# Create the config directory in the binary dir if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/rocm_wmma_gemm/kernel)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/kernel_inst)

# Tell CMake to reconfigure if the config file changes
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/config/gemm_config.json)

# Run the Python script during CMake configuration to generate all files
message(STATUS "Generating GEMM configuration and kernel sources...")
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/config/generate_config.py
            ${CMAKE_CURRENT_SOURCE_DIR}/config/gemm_config.json
            ${CMAKE_CURRENT_BINARY_DIR}/include/rocm_wmma_gemm/kernel/config_generated.hpp
            --kernel-dir ${CMAKE_CURRENT_BINARY_DIR}/src/kernel_inst
    RESULT_VARIABLE GENERATE_RESULT
    OUTPUT_VARIABLE GENERATE_OUTPUT
    ERROR_VARIABLE GENERATE_ERROR
)

if(NOT GENERATE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate configuration: ${GENERATE_ERROR}")
endif()

message(STATUS "${GENERATE_OUTPUT}")

# Include the generated CMake file that lists all kernel sources
set(KERNEL_SOURCES_CMAKE "${CMAKE_CURRENT_BINARY_DIR}/src/kernel_inst/kernel_sources.cmake")
if(EXISTS ${KERNEL_SOURCES_CMAKE})
    include(${KERNEL_SOURCES_CMAKE})
    list(LENGTH KERNEL_INST_SOURCES KERNEL_COUNT)
    message(STATUS "Included kernel_sources.cmake with ${KERNEL_COUNT} kernel sources")
    set(KERNEL_SOURCES ${KERNEL_INST_SOURCES})
else()
    message(FATAL_ERROR "kernel_sources.cmake was not generated")
endif()

# Set compile options for all source files
# if(KERNEL_SOURCES)
#     set_source_files_properties(${KERNEL_SOURCES} PROPERTIES COMPILE_OPTIONS -mcumode)
# endif()

# Glob the generated kernel_lookup.cpp
set(KERNEL_LOOKUP_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/src/kernel_inst/kernel_lookup.cpp")

# Build the library with gemm.cpp, kernel_lookup.cpp, and all generated kernel sources
add_library(rocm_wmma_gemm STATIC src/gemm.cpp ${KERNEL_LOOKUP_SOURCE} ${KERNEL_SOURCES})

# Force CMake to regenerate if kernel sources change
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/src/kernel_inst/kernel_sources.txt)

# Include HIP include directories and generated headers
target_include_directories(rocm_wmma_gemm PUBLIC
    ${HIP_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include  # Add this for generated headers
)

# Link HIP libraries
target_link_libraries(rocm_wmma_gemm PUBLIC ${HIP_LIBRARIES})

# Add optimization flags
target_compile_options(rocm_wmma_gemm PRIVATE
    -O3                 # High optimization
    -ffast-math         # Fast math operations (may break strict IEEE compliance)
    -mcumode            # CU mode for RDNA GPUs
)

# Add tuner executable
add_executable(tuner config/tuner.cpp)

# Include HIP include directories and generated headers
target_include_directories(tuner PUBLIC
    ${HIP_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include  # Add this for generated headers
)

# Link HIP libraries
target_link_libraries(tuner PUBLIC ${HIP_LIBRARIES} hiprtc benchmark::benchmark)

target_compile_definitions(tuner PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

# Copy tune.py to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/tune.py
    ${CMAKE_BINARY_DIR}/tune.py
    COPYONLY
)
