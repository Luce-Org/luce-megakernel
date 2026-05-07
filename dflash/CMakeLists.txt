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
set(CMAKE_INSTALL_RPATH "$ORIGIN/deps/llama.cpp/ggml/src;$ORIGIN/deps/llama.cpp/ggml/src/ggml-cuda;$ORIGIN/../deps/llama.cpp/ggml/src;$ORIGIN/../deps/llama.cpp/ggml/src/ggml-cuda")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ─── ggml (from the llama.cpp submodule) ─────────────────────────────
#
# We use only ggml from the vendored llama.cpp submodule. The drafter is
# loaded via our own custom Qwen3-0.6B forward (qwen3_0p6b_loader.cpp +
# qwen3_0p6b_graph.cpp) rather than libllama, so libllama is not built.
#
# Hardcoded for CUDA. No BLAS, no Metal, no Vulkan, no examples/tests/tools.

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
# Asymmetric KV-quant (Q4_0 K with Q8_0 V etc) needs all fattn quant pairs.
# Build time grows ~3x. Set DFLASH27B_FA_ALL_QUANTS=OFF if you only run
# the spec_prefill demo (target_gen path uses standard quant pairs).
option(DFLASH27B_FA_ALL_QUANTS "Compile ggml-cuda fattn kernels for all KV-quant pairs" ON)
set(GGML_CUDA_FA_ALL_QUANTS ${DFLASH27B_FA_ALL_QUANTS} CACHE BOOL "" FORCE)

# Resolve the CUDA architecture list up-front so downstream logic (notably
# the consumer-Blackwell ggml workaround below) can inspect the actual
# arches nvcc will compile for. The dflash27b target itself is created
# later; its CUDA_ARCHITECTURES property is applied via
# set_target_properties once the target exists.
#
# Turing (75) and Ampere (86) always; Blackwell consumer (120) and Thor
# (110 on CUDA 13+) added when nvcc supports them. DGX Spark /
# GB10 is compute capability 12.1 (121), added at CUDA 12.9+.
if(DFLASH27B_USER_CUDA_ARCHITECTURES)
    set(_dflash27b_archs "${DFLASH27B_USER_CUDA_ARCHITECTURES}")
else()
    set(_dflash27b_archs "75;86")
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

# Consumer Blackwell workaround: skip sm_12x→sm_12xa replacement and FP4
# mmq kernels that can trigger illegal-instruction faults on consumer chips.
# By default, auto-enable when the resolved CUDA arch list includes a 12x
# entry. Set DFLASH27B_USE_BLACKWELL_CONSUMER_FIX=ON to force this behavior
# explicitly (for cross-compiles or custom arch lists).
option(DFLASH27B_USE_BLACKWELL_CONSUMER_FIX
    "Enable ggml consumer-Blackwell workaround (skip sm_12x→sm_12xa, exclude FP4 mmq kernels)" OFF)
if(DFLASH27B_USE_BLACKWELL_CONSUMER_FIX)
    set(_dflash_is_consumer_blackwell ON)
endif()

if(NOT DEFINED _dflash_is_consumer_blackwell)
    set(_dflash_is_consumer_blackwell OFF)
    # Iterate the resolved dflash27b arch list, not raw CMAKE_CUDA_ARCHITECTURES,
    # which is empty on the default path (the project supplies its own list above).
    foreach(_arch IN LISTS _dflash27b_archs)
        string(REGEX REPLACE "[^0-9]" "" _dflash_arch_num "${_arch}")
        if(_dflash_arch_num MATCHES "^12[0-9]$")
            set(_dflash_is_consumer_blackwell ON)
            break()
        endif()
    endforeach()
endif()

if(_dflash_is_consumer_blackwell)
    set(GGML_CUDA_BLACKWELL_CONSUMER ON CACHE BOOL
        "Skip sm_12X→sm_12Xa for consumer Blackwell (no FP4)" FORCE)
endif()
# Use only the ggml subtree of llama.cpp (skip libllama).
add_subdirectory(deps/llama.cpp/ggml EXCLUDE_FROM_ALL)

# The C++ sources include <cuda_runtime.h> directly, so the toolkit headers must
# be available when compiling the library.
find_package(CUDAToolkit REQUIRED)

# ─── dflash27b static library ──────────────────────────────────────

add_library(dflash27b STATIC
    src/errors.cpp
    src/gguf_target_loader.cpp
    src/gguf_draft_loader.cpp
    src/safetensors_draft.cpp
    src/qwen35_target_graph.cpp
    src/qwen3_dflash_graph.cpp
    src/qwen3_drafter.cpp
    src/qwen3_0p6b_loader.cpp
    src/qwen3_0p6b_graph.cpp
    src/flashprefill_q8.cpp
    src/kv_cache.cpp
    src/kv_quant.cpp
    src/delta_net_chunked.cpp
    # Laguna-XS.2 (Poolside) target arch
    src/laguna_target_loader.cpp
    src/laguna_target_graph.cpp
    src/laguna_daemon.cpp
    src/sampler.cpp
)
# FlashPrefill custom CUDA kernels need BF16 WMMA (sm_80+). On Turing (sm_75)
# the drafter uses ggml's flash_attn_ext instead. Guard added after SM check.
# BSA (Block-Sparse Attention) backs the speculative-prefill drafter scoring
# path. Default ON so prefill is fast out of the box. Turn OFF if you don't
# run the spec-prefill stack or are building for sm<80 (cutlass BSA kernels
# need sm_80+; the auto-detect below also disables BSA on legacy arches).
if(NOT DEFINED DFLASH27B_ENABLE_BSA)
    set(DFLASH27B_ENABLE_BSA ON)
endif()

# Apply the arch list resolved above (before add_subdirectory, so the
# consumer-Blackwell workaround can inspect it) to the dflash27b target.
set_target_properties(dflash27b PROPERTIES CUDA_ARCHITECTURES "${_dflash27b_archs}")

# Extract the minimum SM from the arch list so safetensors_draft.cpp can decide
# at compile time whether to convert BF16 draft weights to FP16 (cuBLAS BF16
# GEMM has no tensor core acceleration on SM < 80).
list(GET _dflash27b_archs 0 _dflash27b_min_sm)
# Strip any trailing 'a' suffix (e.g. "121a" → "121")
string(REGEX REPLACE "[^0-9]" "" _dflash27b_min_sm "${_dflash27b_min_sm}")
target_compile_definitions(dflash27b PRIVATE DFLASH27B_MIN_SM=${_dflash27b_min_sm})

# FlashPrefill custom CUDA kernels need BF16 WMMA (sm_80+).
# On sm_75 the drafter falls back to ggml's flash_attn_ext (see qwen3_0p6b_graph.cpp).
if(_dflash27b_min_sm GREATER_EQUAL 80)
    target_sources(dflash27b PRIVATE
        src/flashprefill_kernels.cu
        src/flashprefill_select.cpp
        src/flashprefill.cpp)
    target_compile_definitions(dflash27b PRIVATE DFLASH27B_HAVE_FLASHPREFILL=1)
endif()

# BSA needs sm_80+ (FA-2 derived kernel uses BF16 tensor cores). Auto-disable
# on legacy arches with a clear message instead of failing the link.
if(DFLASH27B_ENABLE_BSA)
    foreach(_arch IN LISTS _dflash27b_archs)
        if(_arch LESS 80)
            message(WARNING
                "DFLASH27B_ENABLE_BSA=ON requested but CUDA_ARCHITECTURES contains '${_arch}' (<80); "
                "disabling BSA (the spec-prefill path will fall back to the WMMA kernel).")
            set(DFLASH27B_ENABLE_BSA OFF)
            break()
        endif()
    endforeach()
endif()
# BSA also requires the Block-Sparse-Attention submodule (with bundled cutlass).
# Existing users who `git pull` without `--recurse-submodules` would otherwise
# hit a missing-header compile error — auto-disable here with a clear message.
if(DFLASH27B_ENABLE_BSA AND NOT EXISTS
   "${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/cutlass/include/cutlass/numeric_types.h")
    message(WARNING
        "DFLASH27B_ENABLE_BSA=ON requested but the Block-Sparse-Attention submodule "
        "is missing or its bundled cutlass is not initialized. Run "
        "`git submodule update --init --recursive` to enable the BSA spec-prefill "
        "path. Disabling BSA for this build.")
    set(DFLASH27B_ENABLE_BSA OFF)
endif()
if(DFLASH27B_ENABLE_BSA)
    target_sources(dflash27b PRIVATE src/bsa_fwd_inst.cu src/bsa_launcher.cu)
endif()

target_include_directories(dflash27b
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDAToolkit_INCLUDE_DIRS}
)
if(DFLASH27B_ENABLE_BSA)
    target_include_directories(dflash27b PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/bsa_stubs
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/Block-Sparse-Attention/csrc/block_sparse_attn/src)
    target_compile_options(dflash27b PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_compile_definitions(dflash27b PRIVATE FLASHATTENTION_DISABLE_DROPOUT FLASH_NAMESPACE=flash DFLASH27B_HAVE_BSA=1)
    # MSVC's <cmath> hides POSIX M_* macros (M_LOG2E etc.) unless _USE_MATH_DEFINES
    # is set before any cmath include. BSA's softmax.h relies on M_LOG2E; define
    # globally on the target so it precedes every TU's first <cmath> include.
    if(WIN32)
        target_compile_definitions(dflash27b PRIVATE _USE_MATH_DEFINES)
    endif()
endif()

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

# PFlash phase-split harness

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/pflash_daemon.cpp")
    add_executable(pflash_daemon test/pflash_daemon.cpp)
    target_include_directories(pflash_daemon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(pflash_daemon PRIVATE dflash27b ggml ggml-cuda)
endif()

# ─── Tests (numerics vs oracle) ────────────────────────────────────

option(DFLASH27B_TESTS "Build numerics tests" ON)
if(DFLASH27B_TESTS)
    if(_dflash27b_min_sm GREATER_EQUAL 80 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_flashprefill_kernels.cpp")
        add_executable(test_flashprefill_kernels test/test_flashprefill_kernels.cpp)
        set_target_properties(test_flashprefill_kernels PROPERTIES CUDA_ARCHITECTURES "${_dflash27b_archs}")
        target_link_libraries(test_flashprefill_kernels PRIVATE dflash27b CUDA::cudart)
    endif()
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
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/spike_thin_copy.cpp")
        add_executable(spike_thin_copy test/spike_thin_copy.cpp)
        target_include_directories(spike_thin_copy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(spike_thin_copy PRIVATE ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_draft_graph.cpp")
        add_executable(smoke_draft_graph test/smoke_draft_graph.cpp)
        target_include_directories(smoke_draft_graph PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_draft_graph PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_qwen3_0p6b_forward.cpp")
        add_executable(smoke_qwen3_0p6b_forward test/smoke_qwen3_0p6b_forward.cpp)
        target_include_directories(smoke_qwen3_0p6b_forward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_qwen3_0p6b_forward PRIVATE dflash27b ggml ggml-cuda)
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
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_load_target_laguna.cpp")
        add_executable(smoke_load_target_laguna test/smoke_load_target_laguna.cpp)
        target_include_directories(smoke_load_target_laguna PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_load_target_laguna PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/smoke_laguna_forward.cpp")
        add_executable(smoke_laguna_forward test/smoke_laguna_forward.cpp)
        target_include_directories(smoke_laguna_forward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(smoke_laguna_forward PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_laguna_ttft.cpp")
        add_executable(bench_laguna_ttft test/bench_laguna_ttft.cpp)
        target_include_directories(bench_laguna_ttft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(bench_laguna_ttft PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_laguna_pflash.cpp")
        add_executable(bench_laguna_pflash test/bench_laguna_pflash.cpp)
        target_include_directories(bench_laguna_pflash PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(bench_laguna_pflash PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/bench_laguna_generate.cpp")
        add_executable(bench_laguna_generate test/bench_laguna_generate.cpp)
        target_include_directories(bench_laguna_generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(bench_laguna_generate PRIVATE dflash27b ggml ggml-cuda)
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_laguna_daemon.cpp")
        add_executable(test_laguna_daemon test/test_laguna_daemon.cpp)
        target_include_directories(test_laguna_daemon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(test_laguna_daemon PRIVATE dflash27b ggml ggml-cuda)
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
