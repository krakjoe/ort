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

/* CUDA kernels for addition */
__global__ void ort_cuda_add_float16_kernel(float16* result, const float16* a, const float16* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = ort_cuda_float16_from_half(
            __hadd(
                ort_cuda_half_from_float16(a[idx]),
                ort_cuda_half_from_float16(b[idx])
            )
        );
    }
}

__global__ void ort_cuda_add_float32_kernel(float32* result, const float32* a, const float32* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void ort_cuda_add_float64_kernel(float64* result, const float64* a, const float64* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void ort_cuda_add_int8_kernel(int8_t* result, const int8_t* a, const int8_t* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void ort_cuda_add_int16_kernel(int16_t* result, const int16_t* a, const int16_t* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void ort_cuda_add_int32_kernel(int32_t* result, const int32_t* a, const int32_t* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void ort_cuda_add_int64_kernel(int64_t* result, const int64_t* a, const int64_t* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void ort_cuda_add_uint8_kernel(uint8_t* result, const uint8_t* a, const uint8_t* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void ort_cuda_add_uint16_kernel(uint16_t* result, const uint16_t* a, const uint16_t* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void ort_cuda_add_uint32_kernel(uint32_t* result, const uint32_t* a, const uint32_t* b, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        result[idx] = a[idx] + b[idx];
    }
}

/* C-linkage wrapper functions */
extern "C" {

void ort_cuda_add_float16(float16* result, const float16* a, const float16* b, size_t count, cudaStream_t stream) {
    ort_cuda_add_float16_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_add_float32(float32* result, const float32* a, const float32* b, size_t count, cudaStream_t stream) {
    ort_cuda_add_float32_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_add_float64(float64* result, const float64* a, const float64* b, size_t count, cudaStream_t stream) {
    ort_cuda_add_float64_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_add_int8(int8_t* result, const int8_t* a, const int8_t* b, size_t count, cudaStream_t stream) {
    ort_cuda_add_int8_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_add_int16(int16_t* result, const int16_t* a, const int16_t* b, size_t count, cudaStream_t stream) {
    ort_cuda_add_int16_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count), 
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_add_int32(int32_t* result, const int32_t* a, const int32_t* b, size_t count, cudaStream_t stream) {
    ort_cuda_add_int32_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_add_int64(int64_t* result, const int64_t* a, const int64_t* b, size_t count, cudaStream_t stream) {
    ort_cuda_add_int64_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_add_uint8(uint8_t* result, const uint8_t* a, const uint8_t* b, size_t count, cudaStream_t stream) {
    ort_cuda_add_uint8_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_add_uint16(uint16_t* result, const uint16_t* a, const uint16_t* b, size_t count, cudaStream_t stream) {
    ort_cuda_add_uint16_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

void ort_cuda_add_uint32(uint32_t* result, const uint32_t* a, const uint32_t* b, size_t count, cudaStream_t stream) {
    ort_cuda_add_uint32_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, count),
        __ort_cuda_threads, 0, stream>>>(
        result, a, b, count
    );
}

} /* extern "C" */