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

#include "maths/backend/cuda/kernels/round.h"

extern ORT_TLS cudaStream_t __ort_cuda_stream;

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, round, float) {
    const float* va = (const float*) a;
    float* res      = (float*)       result;

    if (count * sizeof(float) < __ort_cuda_threshold) {
        goto __ort_math_backend_round_float_relay;
    }

    ort_cuda_round_float(res, va, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_round_float_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, round, FLOAT)
                (res, va, count);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, round, double) {
    const double* va = (const double*) a;
    double* res      = (double*)       result;

    if (count * sizeof(double) < __ort_cuda_threshold) {
        goto __ort_math_backend_round_double_relay;
    }

    ort_cuda_round_double(res, va, count, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_round_double_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, round, DOUBLE)
                (res, va, count);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}