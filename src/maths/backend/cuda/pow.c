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

#include "maths/backend/cuda/kernels/pow.h"  /* CUDA */

extern ORT_TLS cudaStream_t __ort_cuda_stream;

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, pow, float16) {
    const float16* va = (const float16*)a;
    const float16* vb = (const float16*)b;
    float16* res = (float16*)result;

    ort_cuda_pow_float16(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(pow, FLOAT16)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, pow, float32) {
    const float32* va = (const float32*)a;
    const float32* vb = (const float32*)b;
    float32* res = (float32*)result;

    ort_cuda_pow_float32(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(pow, FLOAT32)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, pow, float64) {
    const float64* va = (const float64*)a;
    const float64* vb = (const float64*)b;
    float64* res = (float64*)result;

    ort_cuda_pow_float64(res, va, vb, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
        /* LCOV_EXCL_START */
        ORT_MATH_BACKEND_RELAY_CPU_BINARY(pow, FLOAT64)
            (res, va, vb, count);
        return;
        /* LCOV_EXCL_STOP */
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}
