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
#include <cuda_fp16.h>

#include "maths/backend/cuda/impl.h"

#include "maths/backend/cuda/kernels/matmul.h"

extern ORT_TLS cudaStream_t __ort_cuda_stream;
extern ORT_TLS cublasHandle_t __ort_cublas_handle;

/* Float16 matmul using cublasGemmEx (modern unified interface) */
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, float16) {
    const float16 *va = (const float16 *)a;
    const float16 *vb = (const float16 *)b;
    float16 *res = (float16 *)result;

    if ((a_cols * sizeof(float16) < __ort_cuda_threshold) &&
        (b_cols * sizeof(float16) < __ort_cuda_threshold)) {
        goto __ort_math_backend_matmul_float16_relay;
    }

    /* For row vector a times matrix b using cublasGemmEx:
     * Compute: C = alpha * A * B + beta * C
     * Where A is 1x(a_cols), B is (a_cols)x(b_cols), C is 1x(b_cols)
     */
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        __ort_cublas_handle,
        CUBLAS_OP_N,           /* op(B) = B */
        CUBLAS_OP_N,           /* op(A) = A */
        (int)b_cols,           /* m: rows of op(B) and C */
        1,                     /* n: cols of op(A) and C */
        (int)a_cols,           /* k: cols of op(B) and rows of op(A) */
        &alpha,                /* alpha */
        vb,                    /* B matrix */
        CUDA_R_16F,            /* B data type (half precision) */
        (int)b_cols,           /* ldb: leading dimension of B */
        va,                    /* A vector (treated as 1x(a_cols) matrix) */
        CUDA_R_16F,            /* A data type (half precision) */
        (int)a_cols,           /* lda: leading dimension of A */
        &beta,                 /* beta */
        res,                   /* C result */
        CUDA_R_16F,            /* C data type (half precision) */
        (int)b_cols,           /* ldc: leading dimension of C */
        CUDA_R_32F,            /* computation type (float32 for accuracy) */
        CUBLAS_GEMM_DEFAULT_TENSOR_OP /* algorithm */
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
__ort_math_backend_matmul_float16_relay: // LCOV_EXCL_LINE
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, FLOAT16)
                (res, va, vb, a_cols, b_cols);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

/* Float matmul using cuBLAS gemv (matrix-vector multiply) */
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, float32) {
    const float32 *va = (const float32 *)a;
    const float32 *vb = (const float32 *)b;
    float32 *res = (float32 *)result;

    if ((a_cols * sizeof(float32) < __ort_cuda_threshold) &&
        (b_cols * sizeof(float32) < __ort_cuda_threshold)) {
        goto __ort_math_backend_matmul_float32_relay;
    }

    /* For row vector a times matrix b (both row-major):
     * result[j] = sum_i a[i] * b[i][j]
     * 
     * Pass row-major b to cuBLAS, which interprets it as column-major b^T
     * Then gemv with CUBLAS_OP_N computes: result = b^T * a
     * Which gives us the row vector result we want.
     */
    const float32 alpha = 1.0f;
    const float32 beta = 0.0f;

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
__ort_math_backend_matmul_float32_relay: // LCOV_EXCL_LINE
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, FLOAT32)
                (res, va, vb, a_cols, b_cols);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}

/* Double matmul using cuBLAS gemv */
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, float64) {
    const float64 *va = (const float64 *)a;
    const float64 *vb = (const float64 *)b;
    float64 *res = (float64 *)result;

    if ((a_cols * sizeof(float64) < __ort_cuda_threshold) &&
        (b_cols * sizeof(float64) < __ort_cuda_threshold)) {
        goto __ort_math_backend_matmul_float64_relay;
    }

    const float64 alpha = 1.0;
    const float64 beta = 0.0;

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
__ort_math_backend_matmul_float64_relay: // LCOV_EXCL_LINE
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, FLOAT64)
                (res, va, vb, a_cols, b_cols);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
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
__ort_math_backend_matmul_int8_relay: // LCOV_EXCL_LINE
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, INT8)
                (res, va, vb, a_cols, b_cols);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
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
__ort_math_backend_matmul_uint8_relay: // LCOV_EXCL_LINE
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, UINT8)
                (res, va, vb, a_cols, b_cols);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
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
__ort_math_backend_matmul_int16_relay: // LCOV_EXCL_LINE
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, INT16)
                (res, va, vb, a_cols, b_cols);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
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
__ort_math_backend_matmul_uint16_relay: // LCOV_EXCL_LINE
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, UINT16)
                (res, va, vb, a_cols, b_cols);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
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
__ort_math_backend_matmul_int32_relay: // LCOV_EXCL_LINE
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, INT32)
                (res, va, vb, a_cols, b_cols);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
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
__ort_math_backend_matmul_uint32_relay: // LCOV_EXCL_LINE
        ORT_MATH_BACKEND_RELAY(
            __ort_math_cpu_dispatch, matmul, UINT32)
                (res, va, vb, a_cols, b_cols);
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);
}