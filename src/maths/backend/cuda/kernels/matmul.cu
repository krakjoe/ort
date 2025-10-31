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

/* Matrix-vector multiply kernels for integer types */
/* Each thread computes one output element: result[j] = sum(a[k] * b[k * b_cols + j]) */

__global__ void ort_cuda_matmul_int8_kernel(int8_t* result, const int8_t* a, size_t a_cols, 
                                             const int8_t* b, size_t b_cols) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < b_cols) {
        int32_t sum = 0;
        for (size_t k = 0; k < a_cols; k++) {
            sum += (int32_t)a[k] * (int32_t)b[k * b_cols + j];
        }
        // Clamp to int8_t range
        if (sum > INT8_MAX) {
            sum = INT8_MAX;
        }
        if (sum < INT8_MIN) {
            sum = INT8_MIN;
        }
        result[j] = (int8_t)sum;
    }
}

__global__ void ort_cuda_matmul_uint8_kernel(uint8_t* result, const uint8_t* a, size_t a_cols, 
                                              const uint8_t* b, size_t b_cols) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < b_cols) {
        uint32_t sum = 0;
        for (size_t k = 0; k < a_cols; k++) {
            sum += (uint32_t)a[k] * (uint32_t)b[k * b_cols + j];
        }
        if (sum > UINT8_MAX) {
            sum = UINT8_MAX;
        }
        result[j] = (uint8_t)sum;
    }
}

__global__ void ort_cuda_matmul_int16_kernel(int16_t* result, const int16_t* a, size_t a_cols, 
                                              const int16_t* b, size_t b_cols) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < b_cols) {
        int32_t sum = 0;
        for (size_t k = 0; k < a_cols; k++) {
            sum += (int32_t)a[k] * (int32_t)b[k * b_cols + j];
        }
        if (sum > INT16_MAX) {
            sum = INT16_MAX;
        }
        if (sum < INT16_MIN) {
            sum = INT16_MIN;
        }
        result[j] = (int16_t)sum;
    }
}

__global__ void ort_cuda_matmul_uint16_kernel(uint16_t* result, const uint16_t* a, size_t a_cols, 
                                               const uint16_t* b, size_t b_cols) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < b_cols) {
        uint32_t sum = 0;
        for (size_t k = 0; k < a_cols; k++) {
            sum += (uint32_t)a[k] * (uint32_t)b[k * b_cols + j];
        }
        if (sum > UINT16_MAX) {
            sum = UINT16_MAX;
        }
        result[j] = (uint16_t)sum;
    }
}

__global__ void ort_cuda_matmul_int32_kernel(int32_t* result, const int32_t* a, size_t a_cols, 
                                              const int32_t* b, size_t b_cols) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < b_cols) {
        int32_t sum = 0;
        for (size_t k = 0; k < a_cols; k++) {
            sum += a[k] * b[k * b_cols + j];
        }
        result[j] = sum;
    }
}

__global__ void ort_cuda_matmul_uint32_kernel(uint32_t* result, const uint32_t* a, size_t a_cols, 
                                               const uint32_t* b, size_t b_cols) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < b_cols) {
        uint32_t sum = 0;
        for (size_t k = 0; k < a_cols; k++) {
            sum += a[k] * b[k * b_cols + j];
        }
        result[j] = sum;
    }
}

/* C-linkage wrapper functions */
extern "C" {

void ort_cuda_matmul_int8(int8_t* result, const int8_t* a, size_t a_cols, 
                          const int8_t* b, size_t b_cols, cudaStream_t stream) {
    ort_cuda_matmul_int8_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, b_cols),
        __ort_cuda_threads, 0, stream>>>(
        result, a, a_cols, b, b_cols
    );
}

void ort_cuda_matmul_uint8(uint8_t* result, const uint8_t* a, size_t a_cols, 
                           const uint8_t* b, size_t b_cols, cudaStream_t stream) {
    ort_cuda_matmul_uint8_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, b_cols),
        __ort_cuda_threads, 0, stream>>>(
        result, a, a_cols, b, b_cols
    );
}

void ort_cuda_matmul_int16(int16_t* result, const int16_t* a, size_t a_cols, 
                           const int16_t* b, size_t b_cols, cudaStream_t stream) {
    ort_cuda_matmul_int16_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, b_cols),
        __ort_cuda_threads, 0, stream>>>(
        result, a, a_cols, b, b_cols
    );
}

void ort_cuda_matmul_uint16(uint16_t* result, const uint16_t* a, size_t a_cols, 
                            const uint16_t* b, size_t b_cols, cudaStream_t stream) {
    ort_cuda_matmul_uint16_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, b_cols),
        __ort_cuda_threads, 0, stream>>>(
        result, a, a_cols, b, b_cols
    );
}

void ort_cuda_matmul_int32(int32_t* result, const int32_t* a, size_t a_cols, 
                           const int32_t* b, size_t b_cols, cudaStream_t stream) {
    ort_cuda_matmul_int32_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, b_cols),
        __ort_cuda_threads, 0, stream>>>(
        result, a, a_cols, b, b_cols
    );
}

void ort_cuda_matmul_uint32(uint32_t* result, const uint32_t* a, size_t a_cols, 
                            const uint32_t* b, size_t b_cols, cudaStream_t stream) {
    ort_cuda_matmul_uint32_kernel<<<
        ort_cuda_blocks_count(__ort_cuda_threads, b_cols),
        __ort_cuda_threads, 0, stream>>>(
        result, a, a_cols, b, b_cols
    );
}

} /* extern "C" */