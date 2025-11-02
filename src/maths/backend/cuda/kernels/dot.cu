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

#include <stdint.h>
#include <cuda_runtime.h>

#include "maths/backend/cuda/kernels/util.h"

/* Block reduction helper - sums values in shared memory */
template<typename T>
__device__ void ort_cuda_reduce_sum(T* sdata, int tid) {
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
}

/* CUDA kernel for float dot product */
__global__ void ort_cuda_dot_float_kernel(float* result, const float* a, const float* b, size_t count) {
    extern __shared__ float sdata_f[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    /* Accumulate thread-local sum */
    float sum = 0.0f;
    for (size_t i = idx; i < count; i += stride) {
        sum += a[i] * b[i];
    }
    sdata_f[tid] = sum;

    /* Reduce within block */
    ort_cuda_reduce_sum(sdata_f, tid);

    /* First thread writes block result */
    if (tid == 0) {
        atomicAdd(result, sdata_f[0]);
    }
}

/* CUDA kernel for double dot product */
__global__ void ort_cuda_dot_double_kernel(double* result, const double* a, const double* b, size_t count) {
    extern __shared__ double sdata_d[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (size_t i = idx; i < count; i += stride) {
        sum += a[i] * b[i];
    }
    sdata_d[tid] = sum;

    ort_cuda_reduce_sum(sdata_d, tid);

    if (tid == 0) {
        atomicAdd(result, sdata_d[0]);
    }
}

/* CUDA kernel for int16_t dot product (accumulates in int32_t) */
__global__ void ort_cuda_dot_int16_kernel(int32_t* result, const int16_t* a, const int16_t* b, size_t count) {
    extern __shared__ int32_t sdata_i32[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    int32_t sum = 0;
    for (size_t i = idx; i < count; i += stride) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    sdata_i32[tid] = sum;

    ort_cuda_reduce_sum(sdata_i32, tid);

    if (tid == 0) {
        atomicAdd(result, sdata_i32[0]);
    }
}

/* CUDA kernel for int32_t dot product (accumulates in int64_t) */
__global__ void ort_cuda_dot_int32_kernel(int64_t* result, const int32_t* a, const int32_t* b, size_t count) {
    extern __shared__ int64_t sdata_i64[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    int64_t sum = 0;
    for (size_t i = idx; i < count; i += stride) {
        sum += (int64_t)a[i] * (int64_t)b[i];
    }
    sdata_i64[tid] = sum;

    ort_cuda_reduce_sum(sdata_i64, tid);

    if (tid == 0) {
        atomicAdd((unsigned long long*)result, (unsigned long long)sdata_i64[0]);
    }
}

/* CUDA kernel for uint16_t dot product (accumulates in uint32_t) */
__global__ void ort_cuda_dot_uint16_kernel(uint32_t* result, const uint16_t* a, const uint16_t* b, size_t count) {
    extern __shared__ uint32_t sdata_u32[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    uint32_t sum = 0;
    for (size_t i = idx; i < count; i += stride) {
        sum += (uint32_t)a[i] * (uint32_t)b[i];
    }
    sdata_u32[tid] = sum;

    ort_cuda_reduce_sum(sdata_u32, tid);

    if (tid == 0) {
        atomicAdd(result, sdata_u32[0]);
    }
}

/* CUDA kernel for uint32_t dot product (accumulates in uint64_t) */
__global__ void ort_cuda_dot_uint32_kernel(uint64_t* result, const uint32_t* a, const uint32_t* b, size_t count) {
    extern __shared__ uint64_t sdata_u64[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    uint64_t sum = 0;
    for (size_t i = idx; i < count; i += stride) {
        sum += (uint64_t)a[i] * (uint64_t)b[i];
    }
    sdata_u64[tid] = sum;

    ort_cuda_reduce_sum(sdata_u64, tid);

    if (tid == 0) {
        atomicAdd((unsigned long long*)result, (unsigned long long) sdata_u64[0]);
    }
}

/* C-linkage wrapper functions */
extern "C" {

void ort_cuda_dot_float(float* result, const float* a, const float* b, size_t count, cudaStream_t stream) {
    cudaMemsetAsync(result, 0, sizeof(float), stream);

    ort_cuda_dot_float_kernel<<<
        ort_cuda_blocks_min(__ort_cuda_threads, count, 1024),
        __ort_cuda_threads,
        __ort_cuda_threads * sizeof(float), stream>>>(
        result, a, b, count
    );
}

void ort_cuda_dot_double(double* result, const double* a, const double* b, size_t count, cudaStream_t stream) {
    cudaMemsetAsync(result, 0, sizeof(double), stream);

    ort_cuda_dot_double_kernel<<<
        ort_cuda_blocks_min(__ort_cuda_threads, count, 1024),
        __ort_cuda_threads,
        __ort_cuda_threads * sizeof(double), stream>>>(
        result, a, b, count
    );
}

void ort_cuda_dot_int16(int16_t* result, const int16_t* a, const int16_t* b, size_t count, cudaStream_t stream) {
    // Accumulate in int32_t to avoid overflow
    int32_t temp_result = 0;
    int32_t* d_temp;
    cudaMallocAsync(&d_temp, sizeof(int32_t), stream);
    cudaMemsetAsync(d_temp, 0, sizeof(int32_t), stream);

    ort_cuda_dot_int16_kernel<<<
        ort_cuda_blocks_min(__ort_cuda_threads, count, 1024),
        __ort_cuda_threads,
        __ort_cuda_threads * sizeof(int32_t), stream>>>(
        d_temp, a, b, count
    );

    cudaMemcpyAsync(&temp_result, d_temp, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_temp, stream);

    // Clamp and store result
    if (temp_result > INT16_MAX) {
        temp_result = INT16_MAX;
    }
    if (temp_result < INT16_MIN) {
        temp_result = INT16_MIN;
    }
    *result = (int16_t)temp_result;
}

void ort_cuda_dot_int32(int32_t* result, const int32_t* a, const int32_t* b, size_t count, cudaStream_t stream) {
    int64_t temp_result = 0;
    int64_t* d_temp;
    cudaMallocAsync(&d_temp, sizeof(int64_t), stream);
    cudaMemsetAsync(d_temp, 0, sizeof(int64_t), stream);

    ort_cuda_dot_int32_kernel<<<
        ort_cuda_blocks_min(__ort_cuda_threads, count, 1024),
        __ort_cuda_threads,
        __ort_cuda_threads * sizeof(int64_t), stream>>>(
        d_temp, a, b, count
    );

    cudaMemcpyAsync(&temp_result, d_temp, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_temp, stream);

    // Clamp and store result
    if (temp_result > INT32_MAX) {
        temp_result = INT32_MAX;
    }
    if (temp_result < INT32_MIN) {
        temp_result = INT32_MIN;
    }
    *result = (int32_t)temp_result;
}

void ort_cuda_dot_uint16(uint16_t* result, const uint16_t* a, const uint16_t* b, size_t count, cudaStream_t stream) {
    uint32_t temp_result = 0;
    uint32_t* d_temp;
    cudaMallocAsync(&d_temp, sizeof(uint32_t), stream);
    cudaMemsetAsync(d_temp, 0, sizeof(uint32_t), stream);

    ort_cuda_dot_uint16_kernel<<<
        ort_cuda_blocks_min(__ort_cuda_threads, count, 1024),
        __ort_cuda_threads,
        __ort_cuda_threads * sizeof(uint32_t), stream>>>(
        d_temp, a, b, count
    );

    cudaMemcpyAsync(&temp_result, d_temp, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_temp, stream);

    // Clamp and store result
    if (temp_result > UINT16_MAX) {
        temp_result = UINT16_MAX;
    }

    *result = (uint16_t)temp_result;
}

void ort_cuda_dot_uint32(uint32_t* result, const uint32_t* a, const uint32_t* b, size_t count, cudaStream_t stream) {
    uint64_t temp_result = 0;
    uint64_t* d_temp;
    cudaMallocAsync(&d_temp, sizeof(uint64_t), stream);
    cudaMemsetAsync(d_temp, 0, sizeof(uint64_t), stream);

    ort_cuda_dot_uint32_kernel<<<
        ort_cuda_blocks_min(__ort_cuda_threads, count, 1024),
        __ort_cuda_threads,
        __ort_cuda_threads * sizeof(uint64_t), stream>>>(
        d_temp, a, b, count
    );

    cudaMemcpyAsync(&temp_result, d_temp, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_temp, stream);

    // Clamp and store result
    if (temp_result > UINT32_MAX) {
        temp_result = UINT32_MAX;
    }
    *result = (uint32_t)temp_result;
}

} /* extern "C" */