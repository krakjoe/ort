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

/* CUDA kernel for float16 square root */
__global__ void ort_cuda_sqrt_float16_kernel(float16* result, const float16* a, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        result[idx] = ort_cuda_float16_from_half(
            hsqrt(
                ort_cuda_half_from_float16(a[idx])
            )
        );
    }
}

/* CUDA kernel for float32 square root */
__global__ void ort_cuda_sqrt_float32_kernel(float32* result, const float32* a, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        result[idx] = sqrtf(a[idx]);
    }
}

/* CUDA kernel for float64 square root */
__global__ void ort_cuda_sqrt_float64_kernel(float64* result, const float64* a, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        result[idx] = sqrt(a[idx]);
    }
}

/* C-linkage wrapper functions */
extern "C" {

void ort_cuda_sqrt_float16(float16* result, const float16* a, size_t count, cudaStream_t stream) {
    ort_cuda_sqrt_float16_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, count
    );
}

void ort_cuda_sqrt_float32(float32* result, const float32* a, size_t count, cudaStream_t stream) {
    ort_cuda_sqrt_float32_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, count
    );
}

void ort_cuda_sqrt_float64(float64* result, const float64* a, size_t count, cudaStream_t stream) {
    ort_cuda_sqrt_float64_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, count
    );
}

} /* extern "C" */