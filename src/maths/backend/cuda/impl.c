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

#include "maths/dispatch.h"
#include "maths/backend/cuda/impl.h"

#include <cuda_runtime.h> /* CUDA */
#include <cublas_v2.h>    /* cuBLAS */

static int __ort_cuda_device = 0;

int __ort_cuda_threshold = 16384;

static bool                  __ort_cuda_initialized = false;
ORT_TLS bool                 __ort_cuda_activated = false;
ORT_TLS cudaStream_t         __ort_cuda_stream = NULL;
ORT_TLS cublasHandle_t       __ort_cublas_handle = NULL;
ORT_TLS ort_alloc_t          __ort_math_cpu_allocator;
ORT_TLS ort_math_dispatch_t* __ort_math_cpu_dispatch = NULL;

void ort_cuda_activate(void) {
    if (__ort_cuda_activated) {
        return;
    }

    if (!__ort_cuda_initialized) {
        cudaSetDevice(
            __ort_cuda_device);
        __ort_cuda_initialized = true;
    }

    cudaStreamCreate(&__ort_cuda_stream);
    cublasCreate(&__ort_cublas_handle);
    cublasSetStream(
        __ort_cublas_handle,
        __ort_cuda_stream);
    __ort_cuda_activated = true;
}

static void ort_cuda_deactivate(void) {
    if (!__ort_cuda_activated) {
        return;
    }

    cudaStreamSynchronize(__ort_cuda_stream);

    if (__ort_cublas_handle) {
        cublasDestroy(__ort_cublas_handle);
    }

    cudaStreamDestroy(
        __ort_cuda_stream);
}

static void* ort_cuda_alloc(size_t size, size_t count, size_t alignment) {
    if (size * count < __ort_cuda_threshold) {
        return __ort_math_cpu_allocator.alloc(size, count, alignment);
    }

    ort_cuda_activate();

    void *ptr;
    cudaError_t err = cudaMallocManaged(&ptr, size * count, cudaMemAttachGlobal);

    if (err != cudaSuccess) {
        return NULL;
    }

    if (cudaMemset(ptr, 0, size * count) != cudaSuccess) {
        /* TODO: logging */
    }

    return ptr;
}

static void ort_cuda_free(void* ptr) {
    if (!ptr) {
        return;
    }

    struct cudaPointerAttributes attributes;

    if (__ort_cuda_activated &&
        (cudaPointerGetAttributes(&attributes, ptr) == cudaSuccess &&
        attributes.type == cudaMemoryTypeManaged)) {
        cudaFree(ptr);
        return;
    }

    __ort_math_cpu_allocator.free(ptr);
}

static void* ort_cuda_memcpy(void *dest, const void *src, size_t n) {
    if (n == 0) {
        return dest;
    }

    struct cudaPointerAttributes attributes;

    if (__ort_cuda_activated &&
        (cudaPointerGetAttributes(&attributes, dest) == cudaSuccess &&
        attributes.type == cudaMemoryTypeManaged)) {
        if (cudaMemcpy(dest, src, n, cudaMemcpyDefault) != cudaSuccess) {
            /* TODO: logging */
        }
        return dest;
    }

    return __ort_math_cpu_allocator.memcpy_fn(dest, src, n);
}

void ort_math_backend_gpu_install(ort_math_dispatch_t* table) {
    int devices = 0;

    if (cudaGetDeviceCount(&devices) != cudaSuccess ||
        devices == 0) {
        return;
    }

    /* Set device for this thread from possible ENV */
    const char* env = getenv("ORT_CUDA_DEVICE");

    if (env) {
        /* TODO: pci bus selector */
        __ort_cuda_device = atoi(env);
        /* Check the selected device is reasonable */
        if (__ort_cuda_device >= devices) {
            /* Fallback to the default/first device */
            __ort_cuda_device = 0;
        }
    }

    /* Set threshold for this thread from possible ENV */
    env = getenv("ORT_CUDA_THRESHOLD");

    if (env) {
        __ort_cuda_threshold = atoi(env);
    }

    /* Setup allocator for cuda unified memory */
    ort_alloc_setup(
        &__ort_math_cpu_allocator,
        NULL,
        ort_cuda_alloc,
        ort_cuda_memcpy,
        ort_cuda_free,
        ort_cuda_deactivate
    );

    /* Cuda unified memory is 256 byte aligned */
    ort_alloc_align(256);

    /* Backup CPU dispatch table */
    __ort_math_cpu_dispatch =
        ort_math_dispatch_backup_malloc();

    /* abs.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   abs, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  abs, double)

    /* add.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   add, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  add, double)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT8,    add, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT16,   add, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT32,   add, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT64,   add, int64_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT8,   add, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT16,  add, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT32,  add, uint32_t)

    /* ceil.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   ceil, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  ceil, double)

    /* div.c*/
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   div, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  div, double)

    /* dot.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   dot, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  dot, double)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT16,   dot, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT32,   dot, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT16,  dot, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT32,  dot, uint32_t)

    /* floor.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   floor, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  floor, double)

    /* matmul.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   matmul, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  matmul, double)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT16,   matmul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT32,   matmul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT16,  matmul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT32,  matmul, uint32_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT8,    matmul, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT8,   matmul, uint8_t)

    /* mul.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   mul, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  mul, double)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT16,   mul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT32,   mul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT16,  mul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT32,  mul, uint32_t)

    /* neg.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   neg, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  neg, double)

    /* recip.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   recip, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  recip, double)

    /* round.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   round, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  round, double)

    /* sign.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   sign, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  sign, double)

    /* sqrt.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   sqrt, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  sqrt, double)

    /* sub.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   sub, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  sub, double)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT8,    sub, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT16,   sub, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT32,   sub, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  INT64,   sub, int64_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT8,   sub, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT16,  sub, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  UINT32,  sub, uint32_t)

    /* trunc.c */
    ORT_MATH_BACKEND_INSTALL(table, cuda,  FLOAT,   trunc, float)
    ORT_MATH_BACKEND_INSTALL(table, cuda,  DOUBLE,  trunc, double)
}