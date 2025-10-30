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

#include "maths/backend/cuda/impl.h"

#include "maths/backend/cuda/kernels/mul.h"

extern ORT_TLS cudaStream_t __ort_cuda_stream;

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, int16_t) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;

    if (count * sizeof(int16_t) < __ort_cuda_threshold) {
        goto __ort_math_backend_mul_int16_relay;
    }

    ort_cuda_mul_int16(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_mul_int16_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, mul, INT16)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, int32_t) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;

    if (count * sizeof(int32_t) < __ort_cuda_threshold) {
        goto __ort_math_backend_mul_int32_relay;
    }

    ort_cuda_mul_int32(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_mul_int32_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, mul, INT32)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, uint16_t) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;

    if (count * sizeof(uint16_t) < __ort_cuda_threshold) {
        goto __ort_math_backend_mul_uint16_relay;
    }

    ort_cuda_mul_uint16(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_mul_uint16_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, mul, UINT16)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, uint32_t) {
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;

    if (count * sizeof(uint32_t) < __ort_cuda_threshold) {
        goto __ort_math_backend_mul_uint32_relay;
    }

    ort_cuda_mul_uint32(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_mul_uint32_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, mul, UINT32)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, float) {
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    float* res = (float*)result;

    if (count * sizeof(float) < __ort_cuda_threshold) {
        goto __ort_math_backend_mul_float_relay;
    }

    ort_cuda_mul_float(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_mul_float_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, mul, FLOAT)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, double) {
    const double* va = (const double*)a;
    const double* vb = (const double*)b;
    double* res = (double*)result;

    if (count * sizeof(double) < __ort_cuda_threshold) {
        goto __ort_math_backend_mul_double_relay;
    }

    ort_cuda_mul_double(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_mul_double_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, mul, DOUBLE)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}