cmake_minimum_required(VERSION 3.18)
set(DFLASH27B_USER_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
project(dflash27b LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# If we do not set this, ggml will output to bin and DLLs will not load
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
# Bake portable rpath into all executables so bundled libggml-cuda / libggml-base
# are found regardless of LD_LIBRARY_PATH or stale /usr/local/lib (closes #31).
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/deps/llama.cpp/ggml/src;$ORIGIN/deps/llama.cpp/ggml/src/ggml-cuda;$ORIGIN/../deps/llama.cpp/ggml/src;$ORIGIN/../deps/llama.cpp/ggml/src/ggml-cuda")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

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

# ─── CUDA architecture auto-detection ──────────────────────────────
# CUDA 12.8+ resolves "native" on Blackwell to sm_120a (DGX Spark variant)
# which is forward-incompatible with consumer Blackwell (RTX 5090, SM 12.0).
# Query nvidia-smi at configure time to get the exact SM so ggml-cuda
# compiles for the physical GPU rather than an incompatible variant.
if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    find_program(_DFLASH_NVIDIA_SMI nvidia-smi)
    if(_DFLASH_NVIDIA_SMI)
        execute_process(
            COMMAND "${_DFLASH_NVIDIA_SMI}"
                    --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _dflash_gpu_caps
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _dflash_smi_rc
        )
        if(_dflash_smi_rc EQUAL 0 AND _dflash_gpu_caps)
            string(REGEX MATCH "^[0-9]+\\.[0-9]+" _dflash_cap "${_dflash_gpu_caps}")
            string(REPLACE "." "" _dflash_arch "${_dflash_cap}")
            if(_dflash_arch)
                set(CMAKE_CUDA_ARCHITECTURES "${_dflash_arch}" CACHE STRING
                    "CUDA architectures (auto-detected from nvidia-smi: ${_dflash_cap})" FORCE)
                message(STATUS "dflash27b: GPU compute_cap ${_dflash_cap} → CUDA_ARCHITECTURES=${_dflash_arch}")
                # Consumer Blackwell (SM 12.x, no FP4 tensor cores) must skip
                # ggml's sm_120→sm_120a replacement or kernels will fault.
                if(_dflash_arch MATCHES "^12[0-9]$")
                    set(GGML_CUDA_BLACKWELL_CONSUMER ON CACHE BOOL
                        "Skip sm_12X→sm_12Xa for consumer Blackwell (no FP4)" FORCE)
                endif()
            endif()
        endif()
    endif()
endif()
set(GGML_CUDA_FA_ALL_QUANTS ON CACHE BOOL "Compile fattn kernels for all quant pairs (needed for asymmetric KV-quant)" FORCE)

add_subdirectory(deps/llama.cpp/ggml EXCLUDE_FROM_ALL)

# ─── dflash27b static library ──────────────────────────────────────

add_library(dflash27b STATIC
    src/errors.cpp
    src/gguf_target_loader.cpp
    src/safetensors_draft.cpp
    src/qwen35_target_graph.cpp
    src/qwen3_dflash_graph.cpp
    src/kv_cache.cpp
    src/kv_quant.cpp
    src/f16_convert.cu
    src/delta_net_chunked.cpp
)
# Ampere (86) always; Blackwell consumer (120) and Thor (110 on CUDA 13+)
# added when nvcc supports them. DGX Spark /
# GB10 is compute capability 12.1 (121), added at CUDA 12.9+.
if(DFLASH27B_USER_CUDA_ARCHITECTURES)
    set(_dflash27b_archs "${DFLASH27B_USER_CUDA_ARCHITECTURES}")
else()
    set(_dflash27b_archs "86")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
        list(APPEND _dflash27b_archs "120")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
        list(APPEND _dflash27b_archs "110")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.9")
        list(APPEND _dflash27b_archs "121")
    endif()
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
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_kv_quant.cpp")
        add_executable(test_kv_quant test/test_kv_quant.cpp)
        target_include_directories(test_kv_quant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_kv_quant PRIVATE dflash27b)
    endif()
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
