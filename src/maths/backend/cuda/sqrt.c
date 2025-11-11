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

#include "maths/backend/cuda/kernels/sqrt.h"

extern ORT_TLS cudaStream_t __ort_cuda_stream;

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sqrt, float16) {
    const float16* va = (const float16*) a;
    float16* res      = (float16*)       result;

    ort_cuda_sqrt_float16(res, va, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_UNARY(sqrt, FLOAT16)
            (res, va, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sqrt, float32) {
    const float32* va = (const float32*) a;
    float32* res      = (float32*)       result;

    ort_cuda_sqrt_float32(res, va, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_UNARY(sqrt, FLOAT32)
            (res, va, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sqrt, float64) {
    const float64* va = (const float64*) a;
    float64* res      = (float64*)       result;

    ort_cuda_sqrt_float64(res, va, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_UNARY(sqrt, FLOAT64)
            (res, va, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}