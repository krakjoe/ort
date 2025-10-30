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

#include <cublas_v2.h>

#include "maths/backend/cuda/impl.h"

#include "maths/backend/cuda/kernels/matmul.h"

extern ORT_TLS cudaStream_t __ort_cuda_stream;
extern ORT_TLS cublasHandle_t __ort_cublas_handle;

/* Float matmul using cuBLAS gemv (matrix-vector multiply) */
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, float) {
    const float *va = (const float *)a;
    const float *vb = (const float *)b;
    float *res = (float *)result;

    if ((a_cols * sizeof(float) < __ort_cuda_threshold) &&
        (b_cols * sizeof(float) < __ort_cuda_threshold)) {
        goto __ort_math_backend_matmul_float_relay;
    }

    /* For row vector a times matrix b (both row-major):
     * result[j] = sum_i a[i] * b[i][j]
     * 
     * Pass row-major b to cuBLAS, which interprets it as column-major b^T
     * Then gemv with CUBLAS_OP_N computes: result = b^T * a
     * Which gives us the row vector result we want.
     */
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemv(
        __ort_cublas_handle,
        CUBLAS_OP_N,           /* No transpose needed */
        (int)b_cols,           /* rows of b^T as seen by cuBLAS */
        (int)a_cols,           /* cols of b^T as seen by cuBLAS */
        &alpha,
        vb,                    /* b matrix (row-major) seen as b^T (column-major) */
        (int)b_cols,           /* leading dimension */
        va,                    /* a vector */
        1,                     /* stride */
        &beta,
        res,                   /* result vector */
        1                      /* stride */
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
__ort_math_backend_matmul_float_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, FLOAT)
                (res, va, vb, a_cols, b_cols);
    }
}

/* Double matmul using cuBLAS gemv */
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, double) {
    const double *va = (const double *)a;
    const double *vb = (const double *)b;
    double *res = (double *)result;

    if ((a_cols * sizeof(double) < __ort_cuda_threshold) &&
        (b_cols * sizeof(double) < __ort_cuda_threshold)) {
        goto __ort_math_backend_matmul_double_relay;
    }

    const double alpha = 1.0;
    const double beta = 0.0;

    cublasStatus_t status = cublasDgemv(
        __ort_cublas_handle,
        CUBLAS_OP_N,           /* No transpose needed */
        (int)b_cols,           /* rows of b^T as seen by cuBLAS */
        (int)a_cols,           /* cols of b^T as seen by cuBLAS */
        &alpha,
        vb,                    /* b matrix (row-major) seen as b^T (column-major) */
        (int)b_cols,           /* leading dimension */
        va,                    /* a vector */
        1,                     /* stride */
        &beta,
        res,                   /* result vector */
        1                      /* stride */
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
__ort_math_backend_matmul_double_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, DOUBLE)
                (res, va, vb, a_cols, b_cols);
    }
}

/* Integer matmuls using custom kernels */
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, int8_t) {
    const int8_t *va = (const int8_t *)a;
    const int8_t *vb = (const int8_t *)b;
    int8_t *res = (int8_t *)result;

    if ((a_cols * sizeof(int8_t) < __ort_cuda_threshold) &&
        (b_cols * sizeof(int8_t) < __ort_cuda_threshold)) {
        goto __ort_math_backend_matmul_int8_relay;
    }

    ort_cuda_matmul_int8(res, va, a_cols, vb, b_cols, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_matmul_int8_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, INT8)
                (res, va, vb, a_cols, b_cols);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, uint8_t) {
    const uint8_t *va = (const uint8_t *)a;
    const uint8_t *vb = (const uint8_t *)b;
    uint8_t *res = (uint8_t *)result;

    if ((a_cols * sizeof(uint8_t) < __ort_cuda_threshold) &&
        (b_cols * sizeof(uint8_t) < __ort_cuda_threshold)) {
        goto __ort_math_backend_matmul_uint8_relay;
    }

    ort_cuda_matmul_uint8(res, va, a_cols, vb, b_cols, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_matmul_uint8_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, UINT8)
                (res, va, vb, a_cols, b_cols);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, int16_t) {
    const int16_t *va = (const int16_t *)a;
    const int16_t *vb = (const int16_t *)b;
    int16_t *res = (int16_t *)result;

    if ((a_cols * sizeof(int16_t) < __ort_cuda_threshold) &&
        (b_cols * sizeof(int16_t) < __ort_cuda_threshold)) {
        goto __ort_math_backend_matmul_int16_relay;
    }

    ort_cuda_matmul_int16(res, va, a_cols, vb, b_cols, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_matmul_int16_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, INT16)
                (res, va, vb, a_cols, b_cols);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, uint16_t) {
    const uint16_t *va = (const uint16_t *)a;
    const uint16_t *vb = (const uint16_t *)b;
    uint16_t *res = (uint16_t *)result;

    if ((a_cols * sizeof(uint16_t) < __ort_cuda_threshold) &&
        (b_cols * sizeof(uint16_t) < __ort_cuda_threshold)) {
        goto __ort_math_backend_matmul_uint16_relay;
    }

    ort_cuda_matmul_uint16(res, va, a_cols, vb, b_cols, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_matmul_uint16_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, UINT16)
                (res, va, vb, a_cols, b_cols);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, int32_t) {
    const int32_t *va = (const int32_t *)a;
    const int32_t *vb = (const int32_t *)b;
    int32_t *res = (int32_t *)result;

    if ((a_cols * sizeof(int32_t) < __ort_cuda_threshold) &&
        (b_cols * sizeof(int32_t) < __ort_cuda_threshold)) {
        goto __ort_math_backend_matmul_int32_relay;
    }

    ort_cuda_matmul_int32(res, va, a_cols, vb, b_cols, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_matmul_int32_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, INT32)
                (res, va, vb, a_cols, b_cols);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, uint32_t) {
    const uint32_t *va = (const uint32_t *)a;
    const uint32_t *vb = (const uint32_t *)b;
    uint32_t *res = (uint32_t *)result;

    if ((a_cols * sizeof(uint32_t) < __ort_cuda_threshold) &&
        (b_cols * sizeof(uint32_t) < __ort_cuda_threshold)) {
        goto __ort_math_backend_matmul_uint32_relay;
    }

    ort_cuda_matmul_uint32(res, va, a_cols, vb, b_cols, __ort_cuda_stream);

    if (cudaGetLastError() != cudaSuccess) {
__ort_math_backend_matmul_uint32_relay:
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, UINT32)
                (res, va, vb, a_cols, b_cols);
    }
}