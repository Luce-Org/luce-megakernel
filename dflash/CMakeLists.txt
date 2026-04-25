cmake_minimum_required(VERSION 3.18)
project(dflash27b LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# If we do not set this, ggml will output to bin and DLLs will not load
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ─── ggml (from the llama.cpp submodule's ggml subtree) ─────────────
#
# We pin ggml via llama.cpp's vendored copy for two reasons:
#   1. It's a known-stable ggml version tested against real model code.
#   2. llama.cpp's src/models/{qwen35.cpp,delta-net-base.cpp} sit alongside
#      as READ-ONLY REFERENCE while we port the qwen35 forward pass.
#      They are never built or linked — CMake only consumes the ggml/ subdir.
#
# Hardcoded for CUDA. No libllama, no BLAS, no Metal, no Vulkan.

set(GGML_CUDA         ON  CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL   OFF CACHE BOOL "" FORCE)
set(GGML_METAL        OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN       OFF CACHE BOOL "" FORCE)
set(GGML_BLAS         OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL       OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/ggml/CMakeLists.txt")
    message(FATAL_ERROR
        "deps/llama.cpp submodule missing. Run: "
        "git submodule update --init --recursive")
endif()
add_subdirectory(deps/llama.cpp/ggml EXCLUDE_FROM_ALL)

# ─── dflash27b static library ──────────────────────────────────────

add_library(dflash27b STATIC
    src/errors.cpp
    src/gguf_target_loader.cpp
    src/safetensors_draft.cpp
    src/qwen35_target_graph.cpp
    src/qwen3_dflash_graph.cpp
    src/kv_cache.cpp
    src/f16_convert.cu
    src/delta_net_chunked.cpp
)
# Ampere (86) always; Blackwell consumer (120) added if nvcc supports it
# (CUDA 12.8+), GB10/Jetson Thor (121) added at CUDA 12.9+.
set(_dflash27b_archs "86")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
    list(APPEND _dflash27b_archs "120")
endif()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
    list(APPEND _dflash27b_archs "121")
endif()
set_target_properties(dflash27b PROPERTIES CUDA_ARCHITECTURES "${_dflash27b_archs}")

target_include_directories(dflash27b
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(dflash27b
    PUBLIC
        ggml
        ggml-cuda
        ggml-base
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
     target_compile_options(dflash27b PRIVATE
         $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -Wno-unused-function>
     )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
     target_compile_options(dflash27b PRIVATE
         $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
     )
endif()

# ─── Tests (numerics vs oracle) ────────────────────────────────────

option(DFLASH27B_TESTS "Build numerics tests" ON)
if(DFLASH27B_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_draft_vs_reference.cpp")
        add_executable(test_draft_vs_reference test/test_draft_vs_reference.cpp)
        target_link_libraries(test_draft_vs_reference PRIVATE dflash27b)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_load_draft.cpp")
        add_executable(smoke_load_draft test/smoke_load_draft.cpp)
        target_include_directories(smoke_load_draft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_load_draft PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_draft_graph.cpp")
        add_executable(smoke_draft_graph test/smoke_draft_graph.cpp)
        target_include_directories(smoke_draft_graph PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_draft_graph PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_vs_oracle.cpp")
        add_executable(test_vs_oracle test/test_vs_oracle.cpp)
        target_include_directories(test_vs_oracle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_vs_oracle PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_load_target.cpp")
        add_executable(smoke_load_target test/smoke_load_target.cpp)
        target_include_directories(smoke_load_target PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_load_target PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_target_forward.cpp")
        add_executable(smoke_target_forward test/smoke_target_forward.cpp)
        target_include_directories(smoke_target_forward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_target_forward PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_generate.cpp")
        add_executable(test_generate test/test_generate.cpp)
        target_include_directories(test_generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_generate PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_load_moe_target.cpp")
        add_executable(smoke_load_moe_target test/smoke_load_moe_target.cpp)
        target_include_directories(smoke_load_moe_target PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_load_moe_target PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_dflash.cpp")
        add_executable(test_dflash test/test_dflash.cpp)
        target_include_directories(test_dflash PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_dflash PRIVATE dflash27b ggml ggml-cuda)
        # test_dflash uses cudaMemcpyAsync / cudaMemcpy2DAsync directly for the
        # --fast-rollback path (per-step SSM intermediate state commit). Needs
        # the CUDA runtime on its own link line.
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(test_dflash PRIVATE CUDA::cudart)
        # OpenMP for parallel CPU top-K extraction in the ddtree path.
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(test_dflash PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()
endif()
