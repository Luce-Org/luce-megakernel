/**
 * Portable half-precision type alias for the megakernel.
 *
 * When TARGET_SM >= 80 (Ampere+), uses __nv_bfloat16 (native hardware support).
 * When TARGET_SM < 80  (Turing),   uses __half (fp16) for cuBLAS compatibility.
 *
 * All custom CUDA kernels do explicit H2F/F2H conversions and accumulate in
 * f32, so the choice of 16-bit storage format is transparent to the math.
 */

#pragma once

#ifndef TARGET_SM
#define TARGET_SM 86
#endif

#if TARGET_SM >= 80
  #include <cuda_bf16.h>
  using half_t = __nv_bfloat16;
  #define H2F(x) __bfloat162float(x)
  #define F2H(x) __float2bfloat16(x)
  #define CUBLAS_HALF_T CUDA_R_16BF
#else
  #include <cuda_fp16.h>
  using half_t = __half;
  #define H2F(x) __half2float(x)
  #define F2H(x) __float2half(x)
  #define CUBLAS_HALF_T CUDA_R_16F
#endif
