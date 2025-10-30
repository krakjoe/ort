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

/* CUDA kernel for float division */
__global__ void ort_cuda_div_float_kernel(float* result, const float* a, const float* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        result[idx] = a[idx] / b[idx];
    }
}

/* CUDA kernel for double division */
__global__ void ort_cuda_div_double_kernel(double* result, const double* a, const double* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        result[idx] = a[idx] / b[idx];
    }
}

/* C-linkage wrapper functions */
extern "C" {

void ort_cuda_div_float(float* result, const float* a, const float* b, size_t count, cudaStream_t stream) {
    ort_cuda_div_float_kernel<<<ort_cuda_blocks_count(256, count), 256, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_div_double(double* result, const double* a, const double* b, size_t count, cudaStream_t stream) {
    ort_cuda_div_double_kernel<<<ort_cuda_blocks_count(256, count), 256, 0, stream>>>(
        result, a, b, count
    );
}

} /* extern "C" */