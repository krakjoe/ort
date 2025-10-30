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

#include "maths/backend/cuda/kernels/sub.h"

extern ORT_TLS cudaStream_t __ort_cuda_stream;

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int8_t) {
    const int8_t* va = (const int8_t*)a;
    const int8_t* vb = (const int8_t*)b;
    int8_t* res = (int8_t*)result;

    if (count * sizeof(int8_t) < __ort_cuda_threshold) {
        goto __ort_math_backend_sub_int8_relay;
    }

    ort_cuda_sub_int8(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_sub_int8_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, sub, INT8)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int16_t) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;

    if (count * sizeof(int16_t) < __ort_cuda_threshold) {
        goto __ort_math_backend_sub_int16_relay;
    }

    ort_cuda_sub_int16(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_sub_int16_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, sub, INT16)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int32_t) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;

    if (count * sizeof(int32_t) < __ort_cuda_threshold) {
        goto __ort_math_backend_sub_int32_relay;
    }

    ort_cuda_sub_int32(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_sub_int32_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, sub, INT32)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int64_t) {
    const int64_t* va = (const int64_t*)a;
    const int64_t* vb = (const int64_t*)b;
    int64_t* res = (int64_t*)result;

    if (count * sizeof(int64_t) < __ort_cuda_threshold) {
        goto __ort_math_backend_sub_int64_relay;
    }

    ort_cuda_sub_int64(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_sub_int64_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, sub, INT64)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, uint8_t) {
    const uint8_t* va = (const uint8_t*)a;
    const uint8_t* vb = (const uint8_t*)b;
    uint8_t* res = (uint8_t*)result;

    if (count * sizeof(uint8_t) < __ort_cuda_threshold) {
        goto __ort_math_backend_sub_uint8_relay;
    }

    ort_cuda_sub_uint8(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_sub_uint8_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, sub, UINT8)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, uint16_t) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;

    if (count * sizeof(uint16_t) < __ort_cuda_threshold) {
        goto __ort_math_backend_sub_uint16_relay;
    }

    ort_cuda_sub_uint16(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_sub_uint16_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, sub, UINT16)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, uint32_t) {
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;

    if (count * sizeof(uint32_t) < __ort_cuda_threshold) {
        goto __ort_math_backend_sub_uint32_relay;
    }

    ort_cuda_sub_uint32(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_sub_uint32_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, sub, UINT32)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, float) {
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    float* res = (float*)result;

    if (count * sizeof(float) < __ort_cuda_threshold) {
        goto __ort_math_backend_sub_float_relay;
    }

    ort_cuda_sub_float(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_sub_float_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, sub, FLOAT)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, double) {
    const double* va = (const double*)a;
    const double* vb = (const double*)b;
    double* res = (double*)result;

    if (count * sizeof(double) < __ort_cuda_threshold) {
        goto __ort_math_backend_sub_double_relay;
    }

    ort_cuda_sub_double(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_sub_double_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, sub, DOUBLE)
                (res, va, vb, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}