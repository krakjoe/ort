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

#define ORT_MATH_RESULT_BINARY_IMPL(fname, fdispatch, fschema)              \
ort_tensor_t* ort_math_result_##fname(                                      \
        ort_tensor_t* tensor_a, ort_tensor_t* tensor_b) {                   \
    if (!ort_math_validate_input(tensor_a, #fname) ||                       \
        !ort_math_validate_input(tensor_b, #fname)) {                       \
        return NULL;                                                        \
    }                                                                       \
                                                                            \
    ort_math_promotion_t promotion =                                        \
        ort_math_promotion_perform_binary(fschema, tensor_a, tensor_b);     \
    if (!promotion.is_valid) {                                              \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,            \
            #fname ": incompatible types for operation");                   \
        return NULL;                                                        \
    }                                                                       \
                                                                            \
    ort_math_element_op_func_t operation = fdispatch(&promotion, fschema);  \
    if (!operation) {                                                       \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,            \
            #fname ": unsupported data type for tensor operation");         \
        return NULL;                                                        \
    }                                                                       \
                                                                            \
    return ort_math_result_element_wise_binary(                             \
        &promotion, tensor_a, tensor_b, operation, #fname);                 \
}

#define ORT_MATH_RESULT_SCALAR_IMPL(fname, fdispatch, fschema)              \
ort_tensor_t* ort_math_result_##fname##_scalar(                             \
    ort_tensor_t* tensor, zval* scalar) {                                   \
    if (!ort_math_validate_input(tensor, #fname)) {                         \
        return NULL;                                                        \
    }                                                                       \
    ort_math_promotion_t promotion =                                        \
        ort_math_promotion_perform_scalar(fschema, tensor, scalar);         \
    if (!promotion.is_valid) {                                              \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,            \
            #fname ": incompatible types for operation");                   \
        return NULL;                                                        \
    }                                                                       \
    ort_math_scalar_op_func_t operation = fdispatch(&promotion, fschema);   \
    if (!operation) {                                                       \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,            \
            #fname ": unsupported data type for tensor operation");         \
        return NULL;                                                        \
    }                                                                       \
                                                                            \
    return ort_math_result_element_wise_scalar(                             \
        &promotion, tensor, scalar, operation, #fname);                     \
}

#define ORT_MATH_RESULT_UNARY_IMPL(fname, fdispatch, fschema)             \
ort_tensor_t* ort_math_result_##fname(                                    \
    ort_tensor_t* tensor) {                                               \
    if (!ort_math_validate_input(tensor, #fname)) {                       \
        return NULL;                                                      \
    }                                                                     \
    ort_math_promotion_t promotion =                                      \
        ort_math_promotion_perform_unary(fschema, tensor);                \
    ort_math_unary_op_func_t operation = fdispatch(&promotion, fschema);  \
    if (!operation) {                                                     \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,          \
            #fname ": unsupported data type for mathematical function");  \
        return NULL;                                                      \
    }                                                                     \
    return ort_math_result_element_wise_unary(                            \
        &promotion, tensor, operation, #fname);                           \
}

#define ORT_MATH_RESULT_SERIAL_REDUCE_TENSOR_IMPL(                         \
    fname, fdispatch, fvalidate, fschema)                                  \
ort_tensor_t* ort_math_result_reduce_tensor_##fname(                       \
    ort_tensor_t* tensor) {                                                \
    if (!fvalidate(tensor, #fname)) {                                      \
        return NULL;                                                       \
    }                                                                      \
    ort_math_promotion_t promotion =                                       \
        ort_math_promotion_perform_unary(fschema, tensor);                 \
    ort_math_unary_op_func_t operation = fdispatch(&promotion, fschema);   \
    if (!operation) {                                                      \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,           \
            #fname ": unsupported data type for mathematical function");   \
        return NULL;                                                       \
    }                                                                      \
    return ort_math_result_serial_element_wise_reduce_tensor(              \
        &promotion, tensor, operation, #fname);                            \
}

#define ORT_MATH_RESULT_REDUCE_TENSOR_IMPL(                                 \
    fname, fdispatch, fvalidate, fschema)                                   \
ort_tensor_t* ort_math_result_reduce_tensor_##fname(                        \
    ort_tensor_t* tensor) {                                                 \
    if (!fvalidate(tensor, #fname)) {                                       \
        return NULL;                                                        \
    }                                                                       \
    ort_math_promotion_t promotion =                                        \
        ort_math_promotion_perform_unary(fschema, tensor);                  \
    ort_math_unary_op_func_t operation = fdispatch(&promotion, fschema);    \
    if (!operation) {                                                       \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,            \
            #fname ": unsupported data type for mathematical function");    \
        return NULL;                                                        \
    }                                                                       \
    return ort_math_result_element_wise_reduce_tensor(                      \
        &promotion, tensor, operation, #fname);                             \
}

#define ORT_MATH_RESULT_SERIAL_REDUCE_AXIS_IMPL(                               \
    fname, fdispatch, fvalidate, fvalidatea, fshape, fschema)                  \
ort_tensor_t* ort_math_result_reduce_axis_##fname(                             \
    ort_tensor_t* tensor, zend_long axis, zend_bool keepdims) {                \
    if (!fvalidate(tensor, #fname)) {                                          \
        return NULL;                                                           \
    }                                                                          \
    if (!fvalidatea(tensor, &axis, #fname)) {                                  \
        return NULL;                                                           \
    }                                                                          \
    ort_math_promotion_t promotion =                                           \
        ort_math_promotion_perform_unary(fschema, tensor);                     \
    ort_math_reduction_op_func_t operation = fdispatch(&promotion, fschema);   \
    if (!operation) {                                                          \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,               \
            #fname ": unsupported data type for mathematical function");       \
        return NULL;                                                           \
    }                                                                          \
    return ort_math_result_serial_element_wise_reduce_axis(                    \
        &promotion,                                                            \
        tensor, axis, keepdims,                                                \
        operation, #fname, fshape);                                            \
}

#define ORT_MATH_RESULT_REDUCE_AXIS_IMPL(                                      \
    fname, fdispatch, fvalidate, fvalidatea, fshape, fschema)                  \
ort_tensor_t* ort_math_result_reduce_axis_##fname(                             \
    ort_tensor_t* tensor, zend_long axis, zend_bool keepdims) {                \
    if (!fvalidate(tensor, #fname)) {                                          \
        return NULL;                                                           \
    }                                                                          \
    if (!fvalidatea(tensor, &axis, #fname)) {                                  \
        return NULL;                                                           \
    }                                                                          \
    ort_math_promotion_t promotion =                                           \
        ort_math_promotion_perform_unary(fschema, tensor);                     \
    ort_math_reduction_op_func_t operation = fdispatch(&promotion, fschema);   \
    if (!operation) {                                                          \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,               \
            #fname ": unsupported data type for mathematical function");       \
        return NULL;                                                           \
    }                                                                          \
    return ort_math_result_element_wise_reduce_axis(                           \
        &promotion,                                                            \
        tensor, axis, keepdims,                                                \
        operation, #fname, fshape);                                            \
}

#endif