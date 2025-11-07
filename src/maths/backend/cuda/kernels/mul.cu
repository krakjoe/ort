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
#include <stdint.h>

#include "maths/backend/cuda/kernels/util.h"

/* CUDA kernels for multiplication */
__global__ void ort_cuda_mul_int16_kernel(int16_t* result, const int16_t* a, const int16_t* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void ort_cuda_mul_int32_kernel(int32_t* result, const int32_t* a, const int32_t* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void ort_cuda_mul_uint16_kernel(uint16_t* result, const uint16_t* a, const uint16_t* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void ort_cuda_mul_uint32_kernel(uint32_t* result, const uint32_t* a, const uint32_t* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void ort_cuda_mul_float32_kernel(float32* result, const float32* a, const float32* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void ort_cuda_mul_float64_kernel(float64* result, const float64* a, const float64* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] * b[idx];
    }
}

/* C-linkage wrapper functions */
extern "C" {

void ort_cuda_mul_int16(int16_t* result, const int16_t* a, const int16_t* b, size_t count, cudaStream_t stream) {
    ort_cuda_mul_int16_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_mul_int32(int32_t* result, const int32_t* a, const int32_t* b, size_t count, cudaStream_t stream) {
    ort_cuda_mul_int32_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_mul_uint16(uint16_t* result, const uint16_t* a, const uint16_t* b, size_t count, cudaStream_t stream) {
    ort_cuda_mul_uint16_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_mul_uint32(uint32_t* result, const uint32_t* a, const uint32_t* b, size_t count, cudaStream_t stream) {
    ort_cuda_mul_uint32_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_mul_float32(float32* result, const float32* a, const float32* b, size_t count, cudaStream_t stream) {
    ort_cuda_mul_float32_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_mul_float64(float64* result, const float64* a, const float64* b, size_t count, cudaStream_t stream) {
    ort_cuda_mul_float64_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

} /* extern "C" */