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

#include "maths/backend/cuda/kernels/util.h"

/* CUDA kernel for float floor */
__global__ void ort_cuda_floor_float_kernel(float* result, const float* a, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        result[idx] = floorf(a[idx]);
    }
}

/* CUDA kernel for double floor */
__global__ void ort_cuda_floor_double_kernel(double* result, const double* a, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        result[idx] = floor(a[idx]);
    }
}

/* C-linkage wrapper functions */
extern "C" {

void ort_cuda_floor_float(float* result, const float* a, size_t count, cudaStream_t stream) {
    ort_cuda_floor_float_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, count
    );
}

void ort_cuda_floor_double(double* result, const double* a, size_t count, cudaStream_t stream) {
    ort_cuda_floor_double_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, count
    );
}

} /* extern "C" */