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
#include <stdint.h>

#include "maths/backend/cuda/kernels/util.h"

/* CUDA kernels for modulo */
__global__ void ort_cuda_mod_float16_kernel(float16* result, const float16* a, const float16* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = ort_cuda_float16_from_float32(
            fmodf(
                ort_cuda_float32_from_float16(a[idx]),
                ort_cuda_float32_from_float16(b[idx])
            )
        );
    }
}

__global__ void ort_cuda_mod_float32_kernel(float32* result, const float32* a, const float32* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = fmodf(a[idx], b[idx]);
    }
}

__global__ void ort_cuda_mod_float64_kernel(float64* result, const float64* a, const float64* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = fmod(a[idx], b[idx]);
    }
}

/* C-linkage wrapper functions */
extern "C" {

void ort_cuda_mod_float16(float16* result, const float16* a, const float16* b, size_t count, cudaStream_t stream) {
    ort_cuda_mod_float16_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_mod_float32(float32* result, const float32* a, const float32* b, size_t count, cudaStream_t stream) {
    ort_cuda_mod_float32_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_mod_float64(float64* result, const float64* a, const float64* b, size_t count, cudaStream_t stream) {
    ort_cuda_mod_float64_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}


} /* extern "C" */