# GPU architectures to build arch-specific libraries for.
# Override with -DWMMA_GPU_TARGETS="gfx1100;gfx1151" to change.
set(WMMA_GPU_TARGETS "gfx1100;gfx1151" CACHE STRING "Semicolon-separated list of GPU targets")

foreach(GPU_TARGET ${WMMA_GPU_TARGETS})

    # Per-arch generated directories
    set(ARCH_GEN_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/include_${GPU_TARGET}/rocm_wmma_gemm/kernel)
    set(ARCH_GEN_KERNEL  ${CMAKE_CURRENT_BINARY_DIR}/src/kernel_inst_${GPU_TARGET})

    file(REMOVE_RECURSE ${ARCH_GEN_INCLUDE})
    file(REMOVE_RECURSE ${ARCH_GEN_KERNEL})
    file(MAKE_DIRECTORY ${ARCH_GEN_INCLUDE})
    file(MAKE_DIRECTORY ${ARCH_GEN_KERNEL})

    set(F16_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/config/gemm_config_${GPU_TARGET}_f16.json)
    set(F32_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/config/gemm_config_${GPU_TARGET}_f32.json)

    # Fall back to legacy non-suffixed file if the f16-suffixed one doesn't exist yet
    if(NOT EXISTS ${F16_CONFIG})
        set(F16_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/config/gemm_config_${GPU_TARGET}.json)
    endif()

    set(GENERATE_CMD
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/config/generate_config.py
        --f16-config ${F16_CONFIG}
        ${ARCH_GEN_INCLUDE}/config_generated.hpp
        --kernel-dir ${ARCH_GEN_KERNEL}
    )

    if(EXISTS ${F32_CONFIG})
        list(APPEND GENERATE_CMD --f32-config ${F32_CONFIG})
    endif()

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${F16_CONFIG})
    if(EXISTS ${F32_CONFIG})
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${F32_CONFIG})
    endif()

    message(STATUS "Generating GEMM configuration for ${GPU_TARGET}...")
    execute_process(
        COMMAND ${GENERATE_CMD}
        RESULT_VARIABLE GENERATE_RESULT
        OUTPUT_VARIABLE GENERATE_OUTPUT
        ERROR_VARIABLE  GENERATE_ERROR
    )

    if(NOT GENERATE_RESULT EQUAL 0)
        message(FATAL_ERROR "Code generation failed for ${GPU_TARGET}: ${GENERATE_ERROR}")
    endif()
    message(STATUS "${GENERATE_OUTPUT}")

    set(KERNEL_SOURCES_CMAKE "${ARCH_GEN_KERNEL}/kernel_sources.cmake")
    if(EXISTS ${KERNEL_SOURCES_CMAKE})
        include(${KERNEL_SOURCES_CMAKE})
        # Copy to arch-specific variable immediately — KERNEL_INST_SOURCES is global
        # and would be overwritten by the next loop iteration.
        set(KERNEL_INST_SOURCES_${GPU_TARGET} ${KERNEL_INST_SOURCES})
        list(LENGTH KERNEL_INST_SOURCES_${GPU_TARGET} KERNEL_COUNT)
        message(STATUS "${GPU_TARGET}: ${KERNEL_COUNT} kernel source files")
    else()
        message(FATAL_ERROR "kernel_sources.cmake not generated for ${GPU_TARGET}")
    endif()

    set(KERNEL_LOOKUP_SOURCE "${ARCH_GEN_KERNEL}/kernel_lookup.cpp")

    # Per-arch shared library: librocm_wmma_gemm_<arch>.so
    set(ARCH_LIB rocm_wmma_gemm_${GPU_TARGET})
    add_library(${ARCH_LIB} SHARED
        src/gemm.cpp
        ${KERNEL_LOOKUP_SOURCE}
        ${KERNEL_INST_SOURCES_${GPU_TARGET}}
    )

    target_include_directories(${ARCH_LIB} PUBLIC
        ${HIP_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include_${GPU_TARGET}
    )

    target_link_libraries(${ARCH_LIB} PUBLIC ${HIP_LIBRARIES})

    target_compile_options(${ARCH_LIB} PRIVATE
        --offload-arch=${GPU_TARGET}
        -O3
        -ffast-math
        -mcumode
    )

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${ARCH_GEN_KERNEL}/kernel_sources.txt)

endforeach()

# Header-only interface target for the runtime loader (no arch dependency).
add_library(rocm_wmma_gemm_loader INTERFACE)
target_include_directories(rocm_wmma_gemm_loader INTERFACE
    ${HIP_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(rocm_wmma_gemm_loader INTERFACE ${HIP_LIBRARIES} dl)
# Tell the loader where the arch libraries are at build time.
target_compile_definitions(rocm_wmma_gemm_loader INTERFACE
    ROCM_WMMA_GEMM_LIB_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)

add_executable(tuner config/tuner.cpp)
target_include_directories(tuner PUBLIC
    ${HIP_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(tuner PUBLIC ${HIP_LIBRARIES} hiprtc benchmark::benchmark)
target_link_directories(tuner PRIVATE ${ROCM_ROOT}/lib)
target_compile_definitions(tuner PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    ROCM_INCLUDE_DIR="${ROCM_ROOT}"
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/tune.py
    ${CMAKE_BINARY_DIR}/tune.py
    COPYONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/race.py
    ${CMAKE_BINARY_DIR}/race.py
    COPYONLY
)
