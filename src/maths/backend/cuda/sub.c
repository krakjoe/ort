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

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, float16) {
    const float16* va = (const float16*)a;
    const float16* vb = (const float16*)b;
    float16* res      = (float16*)result;

    ort_cuda_sub_float16(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(sub, FLOAT16)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, float32) {
    const float32* va = (const float32*)a;
    const float32* vb = (const float32*)b;
    float32* res = (float32*)result;

    ort_cuda_sub_float32(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(sub, FLOAT32)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, float64) {
    const float64* va = (const float64*)a;
    const float64* vb = (const float64*)b;
    float64* res = (float64*)result;

    ort_cuda_sub_float64(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(sub, FLOAT64)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int8_t) {
    const int8_t* va = (const int8_t*)a;
    const int8_t* vb = (const int8_t*)b;
    int8_t* res = (int8_t*)result;

    ort_cuda_sub_int8(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(sub, INT8)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int16_t) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;

    ort_cuda_sub_int16(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(sub, INT16)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int32_t) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;

    ort_cuda_sub_int32(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(sub, INT32)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int64_t) {
    const int64_t* va = (const int64_t*)a;
    const int64_t* vb = (const int64_t*)b;
    int64_t* res = (int64_t*)result;

    ort_cuda_sub_int64(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(sub, INT64)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, uint8_t) {
    const uint8_t* va = (const uint8_t*)a;
    const uint8_t* vb = (const uint8_t*)b;
    uint8_t* res = (uint8_t*)result;

    ort_cuda_sub_uint8(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(sub, UINT8)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, uint16_t) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;

    ort_cuda_sub_uint16(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(sub, UINT16)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, uint32_t) {
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;

    ort_cuda_sub_uint32(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(sub, UINT32)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}