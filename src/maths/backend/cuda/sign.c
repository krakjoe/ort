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

#include "maths/backend/cuda/kernels/sign.h"

extern ORT_TLS cudaStream_t __ort_cuda_stream;

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sign, float16) {
    const float16* va = (const float16*) a;
    float16* res      = (float16*)       result;

    if (count * sizeof(float16) < __ort_cuda_threshold) {
        goto __ort_math_backend_sign_float16_relay;
    }

    ort_cuda_sign_float16(res, va, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_sign_float16_relay: // LCOV_EXCL_LINE
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, sign, FLOAT16)
                (res, va, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sign, float32) {
    const float32* va = (const float32*) a;
    float32* res      = (float32*)       result;

    if (count * sizeof(float32) < __ort_cuda_threshold) {
        goto __ort_math_backend_sign_float32_relay;
    }

    ort_cuda_sign_float32(res, va, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_sign_float32_relay: // LCOV_EXCL_LINE
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, sign, FLOAT32)
                (res, va, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sign, float64) {
    const float64* va = (const float64*) a;
    float64* res      = (float64*)       result;

    if (count * sizeof(float64) < __ort_cuda_threshold) {
        goto __ort_math_backend_sign_float64_relay;
    }

    ort_cuda_sign_float64(res, va, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_sign_float64_relay: // LCOV_EXCL_LINE
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, sign, FLOAT64)
                (res, va, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}