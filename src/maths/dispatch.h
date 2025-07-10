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
    ort_math_element_op_func_t add_func;
    ort_math_element_op_func_t sub_func;
    ort_math_element_op_func_t mul_func;
    ort_math_element_op_func_t div_func;
    ort_math_element_op_func_t pow_func;
    ort_math_element_op_func_t mod_func;
    ort_math_element_op_func_t dot_func;
    ort_math_matmul_op_func_t  matmul_func;

    /* Scalar operations */
    ort_math_scalar_op_func_t add_scalar_func;
    ort_math_scalar_op_func_t sub_scalar_func;
    ort_math_scalar_op_func_t mul_scalar_func;
    ort_math_scalar_op_func_t div_scalar_func;
    ort_math_scalar_op_func_t pow_scalar_func;
    ort_math_scalar_op_func_t mod_scalar_func;

    /* Unary operations */
    ort_math_unary_op_func_t neg_func;
    ort_math_unary_op_func_t sqrt_func;
    ort_math_unary_op_func_t sin_func;
    ort_math_unary_op_func_t cos_func;
    ort_math_unary_op_func_t tan_func;
    ort_math_unary_op_func_t asin_func;
    ort_math_unary_op_func_t acos_func;
    ort_math_unary_op_func_t atan_func;
    ort_math_unary_op_func_t sinh_func;
    ort_math_unary_op_func_t cosh_func;
    ort_math_unary_op_func_t tanh_func;
    ort_math_unary_op_func_t log_func;
    ort_math_unary_op_func_t log10_func;
    ort_math_unary_op_func_t exp_func;
    ort_math_unary_op_func_t floor_func;
    ort_math_unary_op_func_t ceil_func;
    ort_math_unary_op_func_t round_func;
    ort_math_unary_op_func_t trunc_func;
    ort_math_unary_op_func_t exp2_func;
    ort_math_unary_op_func_t log2_func;
    ort_math_unary_op_func_t log1p_func;
    ort_math_unary_op_func_t cbrt_func;
    ort_math_unary_op_func_t abs_func;
    ort_math_unary_op_func_t sign_func;
    ort_math_unary_op_func_t recip_func;

    ort_math_unary_op_func_t sum_func;
    ort_math_reduction_op_func_t sum_axis_func;

    ort_math_unary_op_func_t mean_func;
    ort_math_reduction_op_func_t mean_axis_func;

    ort_math_unary_op_func_t min_func;
    ort_math_reduction_op_func_t min_axis_func;

    ort_math_unary_op_func_t max_func;
    ort_math_reduction_op_func_t max_axis_func;

    ort_math_reduction_op_func_t softmax_axis_func;
} ort_math_dispatch_t;

static zend_always_inline int16_t ort_math_dispatch_indexof(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return 0;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return 3;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return 5;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return 6;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return 7;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return 9;

        default:
            return FAILURE;
    }
}

#define ORT_MATH_DISPATCH_INSTALL(table, type, func, symbol) \
    table[ort_math_dispatch_indexof(                         \
        ONNX_TENSOR_ELEMENT_DATA_TYPE_##type                  \
    )].func##_func = symbol;

/* Function dispatch table access */
ort_math_dispatch_t* ort_math_dispatch_table(void);

const ort_math_dispatch_t* ort_math_dispatch_type(
    ONNXTensorElementDataType type);
#endif