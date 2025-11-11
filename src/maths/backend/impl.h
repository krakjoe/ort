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

#ifndef HAVE_ORT_MATHS_BACKEND_IMPL
#define HAVE_ORT_MATHS_BACKEND_IMPL

#include <php.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "alloc.h"
#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/frontend/impl.h"

/**
 * Returns largest multiple of width that is less than or equal to count.
 * 
 * Note: This function must return zero where executing subsequent operations
 *       would cause a buffer overrun or access out of bounds, or other undefined behavior.
 **/
static zend_always_inline size_t ort_math_backend_optimal_count(size_t count, size_t width) {
    if (count < width) {
        return 0; /* Not enough elements for a single operation */
    }

    return (count / width) * width;
}

/* {{{ Each cpu backend must implement this function in its own impl.c */
void ort_math_backend_cpu_install(ort_math_dispatch_t* table); /* }}} */

/* {{{ Each gpu backend must implement these functions in its own impl.c */
void ort_math_backend_gpu_install(ort_math_dispatch_t* table);
void* ort_math_backend_gpu_kernel(
    void* kernel, ONNXTensorElementDataType type,
    ort_tensor_t* result, size_t argc, ...); /* }}} */

/* {{{ 
    CPU backends must call this macro within their install routine 
    to setup the dispatch table.
    This must never be invoked outside of the backend install function. */
#define ORT_MATH_BACKEND_INSTALL(cpu, backend, onnx_type, func, c_type) \
    cpu[ort_math_dispatch_indexof(                                      \
        ONNX_TENSOR_ELEMENT_DATA_TYPE_##onnx_type                       \
    )].func##_func = ORT_MATH_DISPATCH_TAG_CPU(                         \
        ORT_MATH_BACKEND_OP_SYMBOL(backend, func, c_type)); /* }}} */

/* {{{ 
    GPU backends must call this macro within their install routine 
    to setup the dispatch table with GPU tagging.
    This must never be invoked outside of the backend install function. */
#define ORT_MATH_BACKEND_INSTALL_GPU(cpu, gpu, backend, onnx_type, func, c_type) do {\
    cpu[ort_math_dispatch_indexof(                               \
        ONNX_TENSOR_ELEMENT_DATA_TYPE_##onnx_type                \
    )].func##_func = ORT_MATH_DISPATCH_TAG_GPU(                  \
        ORT_MATH_DISPATCH_UNTAG(                                 \
            cpu[ort_math_dispatch_indexof(                       \
                ONNX_TENSOR_ELEMENT_DATA_TYPE_##onnx_type        \
            )].func##_func),                                     \
        func##_func);                                            \
    gpu[ort_math_dispatch_indexof(                               \
        ONNX_TENSOR_ELEMENT_DATA_TYPE_##onnx_type                \
    )].func##_func = ORT_MATH_DISPATCH_TAG_GPU(                  \
        ORT_MATH_BACKEND_OP_SYMBOL(backend, func, c_type),       \
            func##_func);                                        \
} while (0);
/* }}} */

/* {{{
    GPU backends must call this (or derivative) macro to relay to CPU backend */
#define ORT_MATH_BACKEND_RELAY_CPU(func, onnx_type, c_type)   \
    ((c_type) ORT_MATH_DISPATCH_UNTAG(                    \
        (ort_math_dispatch_table())[                      \
            ort_math_dispatch_indexof(                    \
                ONNX_TENSOR_ELEMENT_DATA_TYPE_##onnx_type \
            )                                             \
    ].func##_func))

#define ORT_MATH_BACKEND_RELAY_CPU_BINARY(func, onnx_type) \
    ORT_MATH_BACKEND_RELAY_CPU(func, onnx_type, ort_math_kernel_binary_t)
#define ORT_MATH_BACKEND_RELAY_CPU_UNARY(func, onnx_type) \
    ORT_MATH_BACKEND_RELAY_CPU(func, onnx_type, ort_math_kernel_unary_t)
#define ORT_MATH_BACKEND_RELAY_CPU_MATMUL(func, onnx_type) \
    ORT_MATH_BACKEND_RELAY_CPU(func, onnx_type, ort_math_kernel_matmul_t) /* }}} */
#endif