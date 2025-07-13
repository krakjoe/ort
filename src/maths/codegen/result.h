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

/**
 * @file codegen/result.h
 * @brief Code generation macros for mathematical operation results.
 */

#ifndef HAVE_ORT_MATHS_CODEGEN_RESULT
#define HAVE_ORT_MATHS_CODEGEN_RESULT

#include "maths/promotion.h"
#include "maths/result.h"
#include "maths/validate.h"

#include "maths/codegen/common.h"

#define ORT_MATH_RESULT_BINARY_IMPL(func_name, getter_func, schema) \
ort_tensor_t* ort_math_result_##func_name(                              \
        ort_tensor_t* tensor_a, ort_tensor_t* tensor_b) {               \
    if (!ort_math_validate_input(tensor_a, #func_name) ||               \
        !ort_math_validate_input(tensor_b, #func_name)) {               \
        return NULL;                                                    \
    }                                                                   \
                                                                        \
    ort_math_promotion_t promotion =                               \
        ort_math_promotion_perform_binary(schema, tensor_a, tensor_b); \
    if (!promotion.is_valid) {                                          \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,        \
            #func_name ": incompatible types for operation");           \
        return NULL;                                                    \
    }                                                                   \
                                                                        \
    ort_math_element_op_func_t op_func =                                \
        getter_func(promotion.result_type);                             \
    if (!op_func) {                                                     \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,        \
            #func_name ": unsupported data type for tensor operation"); \
        return NULL;                                                    \
    }                                                                   \
                                                                        \
    return ort_math_result_element_wise_binary(                         \
        &promotion, tensor_a, tensor_b, op_func, #func_name);           \
}

#define ORT_MATH_RESULT_SCALAR_IMPL(func_name, getter_func, schema) \
ort_tensor_t* ort_math_result_##func_name##_scalar(                     \
    ort_tensor_t* tensor, zval* scalar) {                               \
    if (!ort_math_validate_input(tensor, #func_name)) {                 \
        return NULL;                                                    \
    }                                                                   \
    ort_math_promotion_t promotion =                               \
        ort_math_promotion_perform_scalar(schema, tensor, scalar);    \
    if (!promotion.is_valid) {                                          \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,        \
            #func_name ": incompatible types for operation");           \
        return NULL;                                                    \
    }                                                                   \
    ort_math_scalar_op_func_t op_func = getter_func(promotion.result_type); \
    if (!op_func) {                                                     \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,        \
            #func_name ": unsupported data type for tensor operation"); \
        return NULL;                                                    \
    }                                                                   \
                                                                        \
    return ort_math_result_element_wise_scalar(                         \
        &promotion, tensor, scalar, op_func, #func_name);               \
}

#define ORT_MATH_RESULT_UNARY_IMPL(func_name, getter_func, schema)           \
ort_tensor_t* ort_math_result_##func_name(                                   \
    ort_tensor_t* tensor) {                                                  \
    if (!ort_math_validate_input(tensor, #func_name)) {                      \
        return NULL;                                                         \
    }                                                                        \
    ort_math_promotion_t promotion =                                    \
        ort_math_promotion_perform_unary(schema, tensor);                  \
    ort_math_unary_op_func_t op_func = getter_func(promotion.result_type);   \
    if (!op_func) {                                                          \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,             \
            #func_name ": unsupported data type for mathematical function"); \
        return NULL;                                                         \
    }                                                                        \
    return ort_math_result_element_wise_unary(&promotion, tensor, op_func, #func_name);  \
}

#define ORT_MATH_RESULT_REDUCE_TENSOR_IMPL(                                    \
    func_name, getter_func, validate_tensor_func, schema)                      \
ort_tensor_t* ort_math_result_reduce_tensor_##func_name(                       \
    ort_tensor_t* tensor) {                                                    \
    if (!validate_tensor_func(tensor, #func_name)) {                           \
        return NULL;                                                           \
    }                                                                          \
    ort_math_promotion_t promotion =                                      \
        ort_math_promotion_perform_unary(schema, tensor);                    \
    ort_math_unary_op_func_t op_func = getter_func(promotion.result_type);     \
    if (!op_func) {                                                            \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,               \
            #func_name ": unsupported data type for mathematical function");   \
        return NULL;                                                           \
    }                                                                          \
    return ort_math_result_element_wise_reduce_tensor(                         \
        &promotion, tensor, op_func, #func_name);                              \
}

#define ORT_MATH_RESULT_SERIAL_REDUCE_AXIS_IMPL(                   \
    func_name, getter_func, validate_tensor_func,validate_axis_func, shape_func, schema) \
ort_tensor_t* ort_math_result_reduce_axis_##func_name(                         \
    ort_tensor_t* tensor, zend_long axis, zend_bool keepdims) {                \
    if (!validate_tensor_func(tensor, #func_name)) {                           \
        return NULL;                                                           \
    }                                                                          \
    if (!validate_axis_func(tensor, &axis, #func_name)) {                      \
        return NULL;                                                           \
    }                                                                          \
    ort_math_promotion_t promotion =                                      \
        ort_math_promotion_perform_unary(schema, tensor);                    \
    ort_math_reduction_op_func_t op_func = getter_func(promotion.result_type); \
    if (!op_func) {                                                            \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,               \
            #func_name ": unsupported data type for mathematical function");   \
        return NULL;                                                           \
    }                                                                          \
    return ort_math_result_serial_element_wise_reduce_axis(                    \
        &promotion,                                                            \
        tensor, axis, keepdims,                                                \
        op_func, #func_name, shape_func);                                      \
}

#define ORT_MATH_RESULT_REDUCE_AXIS_IMPL(                          \
    func_name, getter_func, validate_tensor_func,validate_axis_func, shape_func, schema) \
ort_tensor_t* ort_math_result_reduce_axis_##func_name(                         \
    ort_tensor_t* tensor, zend_long axis, zend_bool keepdims) {                \
    if (!validate_tensor_func(tensor, #func_name)) {                           \
        return NULL;                                                           \
    }                                                                          \
    if (!validate_axis_func(tensor, &axis, #func_name)) {                      \
        return NULL;                                                           \
    }                                                                          \
    ort_math_promotion_t promotion =                                      \
        ort_math_promotion_perform_unary(schema, tensor);                    \
    ort_math_reduction_op_func_t op_func = getter_func(promotion.result_type); \
    if (!op_func) {                                                            \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,               \
            #func_name ": unsupported data type for mathematical function");   \
        return NULL;                                                           \
    }                                                                          \
    return ort_math_result_element_wise_reduce_axis(                           \
        &promotion,                                                            \
        tensor, axis, keepdims,                                                \
        op_func, #func_name, shape_func);                                      \
}

#endif