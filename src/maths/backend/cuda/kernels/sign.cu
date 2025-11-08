/*
  +----------------------------------------------------------------------+
  | ort                                                                  |
  +----------------------------------------------------------------------+
  | Copyright (c) Joe Watkins 2025                                       |
  +----------------------------------------------------------------------+
  | This source file is subject to version 3.01 of the PHP license,      |
  | that is bundled with this package in the file LICENSE, and is        |
  | available through the world-wide-web at the following url:           |
  | http://www.php.net/license/3_01.txt                                  |
  | If you did not receive a copy of the PHP license and are unable to   |
  | obtain it through the world-wide-web, please send a note to          |
  | license@php.net so we can mail you a copy immediately.               |
  +----------------------------------------------------------------------+
  | Author: krakjoe                                                      |
  +----------------------------------------------------------------------+
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "maths/backend/cuda/kernels/util.h"

/* CUDA kernel for float16 sign */
__global__ void ort_cuda_sign_float16_kernel(float16* result, const float16* a, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        __half val = ort_cuda_half_from_float16(a[idx]);
        __half zero = __float2half(0.0f);
        if (__hgt(val, zero)) {
            result[idx] = ort_cuda_float16_from_half(
                __float2half(1.0f)
            );
        } else if (__hlt(val, zero)) {
            result[idx] = ort_cuda_float16_from_half(
                __float2half(-1.0f)
            );
        } else {
            result[idx] = ort_cuda_float16_from_half(zero);
        }
    }
}

/* CUDA kernel for float32 sign */
__global__ void ort_cuda_sign_float32_kernel(float32* result, const float32* a, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        float32 val = a[idx];
        if (val > 0.0f) {
            result[idx] = 1.0f;
        } else if (val < 0.0f) {
            result[idx] = -1.0f;
        } else {
            result[idx] = 0.0f;
        }
    }
}

/* CUDA kernel for float64 sign */
__global__ void ort_cuda_sign_float64_kernel(float64* result, const float64* a, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        float64 val = a[idx];
        if (val > 0.0) {
            result[idx] = 1.0;
        } else if (val < 0.0) {
            result[idx] = -1.0;
        } else {
            result[idx] = 0.0;
        }
    }
}

/* C-linkage wrapper functions */
extern "C" {

void ort_cuda_sign_float16(float16* result, const float16* a, size_t count, cudaStream_t stream) {
    ort_cuda_sign_float16_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, count
    );
}

void ort_cuda_sign_float32(float32* result, const float32* a, size_t count, cudaStream_t stream) {
    ort_cuda_sign_float32_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, count
    );
}

void ort_cuda_sign_float64(float64* result, const float64* a, size_t count, cudaStream_t stream) {
    ort_cuda_sign_float64_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, count
    );
}

} /* extern "C" */