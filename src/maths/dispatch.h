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

#ifndef HAVE_ORT_MATHS_DISPATCH
#define HAVE_ORT_MATHS_DISPATCH

#include "maths/core.h"

/* Function dispatch table structure for mathematical operations */
typedef struct _ort_math_dispatch_t {
    ONNXTensorElementDataType type;

    /* Binary element-wise operations */
    ort_math_kernel_binary_t add_func;
    ort_math_kernel_binary_t sub_func;
    ort_math_kernel_binary_t mul_func;
    ort_math_kernel_binary_t div_func;
    ort_math_kernel_binary_t pow_func;
    ort_math_kernel_binary_t mod_func;
    ort_math_kernel_binary_t dot_func;
    ort_math_kernel_matmul_t matmul_func;

    /* Scalar operations */
    ort_math_kernel_scalar_t add_scalar_func;
    ort_math_kernel_scalar_t sub_scalar_func;
    ort_math_kernel_scalar_t mul_scalar_func;
    ort_math_kernel_scalar_t div_scalar_func;
    ort_math_kernel_scalar_t pow_scalar_func;
    ort_math_kernel_scalar_t mod_scalar_func;

    /* Unary operations */
    ort_math_kernel_unary_t neg_func;
    ort_math_kernel_unary_t sqrt_func;
    ort_math_kernel_unary_t sin_func;
    ort_math_kernel_unary_t cos_func;
    ort_math_kernel_unary_t tan_func;
    ort_math_kernel_unary_t asin_func;
    ort_math_kernel_unary_t acos_func;
    ort_math_kernel_unary_t atan_func;
    ort_math_kernel_unary_t sinh_func;
    ort_math_kernel_unary_t cosh_func;
    ort_math_kernel_unary_t tanh_func;
    ort_math_kernel_unary_t log_func;
    ort_math_kernel_unary_t log10_func;
    ort_math_kernel_unary_t exp_func;
    ort_math_kernel_unary_t floor_func;
    ort_math_kernel_unary_t ceil_func;
    ort_math_kernel_unary_t round_func;
    ort_math_kernel_unary_t trunc_func;
    ort_math_kernel_unary_t exp2_func;
    ort_math_kernel_unary_t log2_func;
    ort_math_kernel_unary_t log1p_func;
    ort_math_kernel_unary_t cbrt_func;
    ort_math_kernel_unary_t abs_func;
    ort_math_kernel_unary_t sign_func;
    ort_math_kernel_unary_t recip_func;

    ort_math_kernel_reduce_tensor_t sum_func;
    ort_math_kernel_reduce_axis_t sum_axis_func;

    ort_math_kernel_reduce_tensor_t mean_func;
    ort_math_kernel_reduce_axis_t mean_axis_func;

    ort_math_kernel_reduce_tensor_t min_func;
    ort_math_kernel_reduce_axis_t min_axis_func;

    ort_math_kernel_reduce_tensor_t max_func;
    ort_math_kernel_reduce_axis_t max_axis_func;

    ort_math_kernel_reduce_axis_t softmax_axis_func;

    ort_math_kernel_reduce_tensor_t argmax_func;
    ort_math_kernel_reduce_axis_t argmax_axis_func;
} ort_math_dispatch_t;

static zend_always_inline int16_t ort_math_dispatch_indexof(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 0;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32: return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return 3;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return 5;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return 6;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return 7;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return 9;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return 10;

        default:
            return FAILURE;
    }
}

/* Function dispatch table access */
ort_math_dispatch_t* ort_math_dispatch_table(void);

/* Function dispatch table backup */
ort_math_dispatch_t*
    ort_math_dispatch_backup_malloc(void);
void ort_math_dispatch_backup_free(
    ort_math_dispatch_t* backup);

const ort_math_dispatch_t* ort_math_dispatch_type(
    ONNXTensorElementDataType type);

/* {{{ kernel tagging */
#define __ORT_MATH_DISPATCH_TAG_CPU      0x0
#define __ORT_MATH_DISPATCH_TAG_GPU      0x1
#define __ORT_MATH_DISPATCH_TAG_MASK     0x1
#define __ORT_MATH_DISPATCH_OFFSET_MASK  0xFE
#define __ORT_MATH_DISPATCH_OFFSET_SHIFT 1

#define ORT_MATH_DISPATCH_OFFSETOF(field)    \
    ((offsetof(ort_math_dispatch_t, field) - \
      offsetof(ort_math_dispatch_t, add_func)) / \
        sizeof(void*))

#define ORT_MATH_DISPATCH_TAG_CPU(kernel) \
    ((void*)                              \
        (((uintptr_t)(kernel)) |          \
        ((__ORT_MATH_DISPATCH_TAG_CPU) &  \
            __ORT_MATH_DISPATCH_TAG_MASK)))

#define ORT_MATH_DISPATCH_TAG_GPU(kernel, field) \
    ((void*)(                                    \
        ((uintptr_t)(kernel)) |                  \
        ((__ORT_MATH_DISPATCH_TAG_GPU) &         \
            __ORT_MATH_DISPATCH_TAG_MASK) |      \
        ((ORT_MATH_DISPATCH_OFFSETOF(field) <<   \
            __ORT_MATH_DISPATCH_OFFSET_SHIFT) &  \
                __ORT_MATH_DISPATCH_OFFSET_MASK)))

#define ORT_MATH_DISPATCH_UNTAG(kernel)      \
    ((void*)(((uintptr_t)(kernel)) &         \
        ~(__ORT_MATH_DISPATCH_TAG_MASK |     \
            __ORT_MATH_DISPATCH_OFFSET_MASK)))

#define ORT_MATH_DISPATCH_TAGGED(kernel, tag) \
    ((((uintptr_t)(kernel)) &                 \
        __ORT_MATH_DISPATCH_TAG_MASK) ==      \
            __ORT_MATH_DISPATCH_TAG_##tag)    \

#define ORT_MATH_DISPATCH_OFFSET(kernel)    \
    ((((uintptr_t)(kernel)) &               \
        __ORT_MATH_DISPATCH_OFFSET_MASK) >> \
            __ORT_MATH_DISPATCH_OFFSET_SHIFT)

static inline void* __ort_math_dispatch_resolve(
    ort_math_dispatch_t* table,
    void* kernel, ONNXTensorElementDataType type) {
    size_t idx =
        ORT_MATH_DISPATCH_OFFSET(kernel);
    ort_math_dispatch_t* dispatch =
        &table[ort_math_dispatch_indexof(type)];
    void** dispatching =
        (void**)
            &dispatch->add_func;
    return dispatching[idx];
}

#define ORT_MATH_DISPATCH_RESOLVE(table, kernel, type) __ort_math_dispatch_resolve(table, kernel, type)
/* }}} */
#endif