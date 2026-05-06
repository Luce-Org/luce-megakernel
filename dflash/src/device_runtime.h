#pragma once

#if defined(DFLASH27B_USE_HIP)
#if defined(__HIPCC__)
#include "../deps/llama.cpp/ggml/src/ggml-cuda/vendors/hip.h"
using __nv_bfloat16 = __hip_bfloat16;
#ifndef cudaEventCreate
#define cudaEventCreate hipEventCreate
#endif
#ifndef cudaEventElapsedTime
#define cudaEventElapsedTime hipEventElapsedTime
#endif
#ifndef __ballot_sync
#define __ballot_sync(mask, predicate) __ballot(predicate)
#endif
#else
#include <hip/hip_runtime.h>

#define cudaEvent_t hipEvent_t
#define cudaStream_t hipStream_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetDevice hipGetDevice
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDestroy hipEventDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#endif
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif
