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

#ifndef HAVE_ORT_MATHS_CORE
#define HAVE_ORT_MATHS_CORE

/* @file core.h
 * @brief Core math API for implementation of mathematical operations.
 * 
 * Note: This file is intended for use at the C layer, and is not
 *       intended for use at the PHP layer.
 */
#include "ort.h"
#include "tensor.h"

/* Mathematical operation result structure */
typedef struct _ort_math_result_t {
    ort_tensor_t* tensor;
    zend_bool success;
    zend_string* error_message;
} ort_math_result_t;

/* Broadcasting information */
typedef struct _ort_math_broadcast_info_t {
    int64_t* result_shape;
    size_t result_dimensions;
    zend_bool is_compatible;
    zend_bool needs_broadcast_a;
    zend_bool needs_broadcast_b;
} ort_math_broadcast_info_t;

/* Type promotion rules */
typedef struct _ort_math_type_promotion_t {
    ONNXTensorElementDataType result_type;
    zend_bool is_valid;
} ort_math_type_promotion_t;

/* Core mathematical operation function pointers */
typedef void (*ort_math_element_op_func_t)(void* result, const void* a, const void* b, size_t count);
typedef void (*ort_math_scalar_op_func_t)(void* result, const void* a, const void* b, size_t count);
typedef void (*ort_math_unary_op_func_t)(void* result, const void* input, size_t count);

/* Broadcasting functions */
ort_math_broadcast_info_t* ort_math_broadcast_calculate(
    const int64_t* shape_a, size_t dims_a,
    const int64_t* shape_b, size_t dims_b
);

void ort_math_broadcast_free(ort_math_broadcast_info_t* info);

/* Type promotion functions */
ort_math_type_promotion_t ort_math_type_promote(
    ONNXTensorElementDataType type_a,
    ONNXTensorElementDataType type_b
);

ort_math_type_promotion_t ort_math_type_promote_strict(
    ONNXTensorElementDataType type_a,
    ONNXTensorElementDataType type_b
);

/* Shape comparison helper */
zend_bool ort_math_shapes_identical(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);

/* Stack allocation threshold for math optimizations */
#define ORT_MATH_MAX_STACK_DIMENSIONS 8

/* Element-wise binary operation */
ort_math_result_t* ort_math_result_element_wise_binary(
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b,
    ort_math_element_op_func_t operation,
    const char* operation_name
);

/* Fast path for same-shape operations */
ort_math_result_t* ort_math_result_element_wise_binary_fast(
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b,
    ort_math_element_op_func_t operation,
    const char* operation_name
);

/* Element-wise scalar operation */
ort_math_result_t* ort_math_result_element_wise_scalar(
    ort_tensor_t* tensor,
    zval* scalar,
    ort_math_scalar_op_func_t operation,
    const char* operation_name
);

/* Element-wise unary operation */
ort_math_result_t* ort_math_result_element_wise_unary(
    ort_tensor_t* tensor,
    ort_math_unary_op_func_t operation,
    const char* operation_name
);

/* Result management */
ort_math_result_t* ort_math_result_create(ort_tensor_t* tensor);
void ort_math_result_free(ort_math_result_t* result);

/* Shape manipulation helpers */
ort_tensor_t* ort_tensor_create_result(
    const int64_t* shape,
    size_t dimensions,
    ONNXTensorElementDataType type,
    const char* name_prefix
);

/* Memory management for mathematical operations */
void* ort_math_memory_alloc_aligned(size_t size, size_t alignment);
void ort_math_memory_free_aligned(void* ptr);

/* Validation functions */
zend_bool ort_math_validate_input(ort_tensor_t* tensor, const char* operation_name);
zend_bool ort_math_validate_axis(ort_tensor_t* tensor, zend_long axis, const char* operation_name);

/* Utility functions */
size_t ort_math_util_calculate_total_elements(const int64_t* shape, size_t dimensions);
zend_long ort_math_util_get_flat_index(const int64_t* indices, const int64_t* shape, size_t dimensions);
void ort_math_util_get_multi_index(zend_long flat_index, const int64_t* shape, size_t dimensions, int64_t* indices);
void ort_math_cast_element(const void* src, void* dst, ONNXTensorElementDataType src_type, ONNXTensorElementDataType dst_type);

/* Function dispatch table structure for mathematical operations */
typedef struct _ort_math_type_dispatch_t {
    ONNXTensorElementDataType type;

    /* Binary element-wise operations */
    ort_math_element_op_func_t add_func;
    ort_math_element_op_func_t sub_func;
    ort_math_element_op_func_t mul_func;
    ort_math_element_op_func_t div_func;
    ort_math_element_op_func_t pow_func;
    ort_math_element_op_func_t mod_func;
    ort_math_element_op_func_t dot_func;

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
} ort_math_type_dispatch_t;

static zend_always_inline int16_t ort_math_type_dispatch_indexof(ONNXTensorElementDataType type) {
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

/* Function dispatch table access */
const ort_math_type_dispatch_t* ort_math_get_dispatch(ONNXTensorElementDataType type);

/* Memory management functions */
void ort_math_broadcast_info_destroy(ort_math_broadcast_info_t* info);
void ort_math_result_destroy(ort_math_result_t* result);
#endif
