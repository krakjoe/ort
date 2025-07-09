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

#ifndef HAVE_ORT_MATHS_CODEGEN
#define HAVE_ORT_MATHS_CODEGEN

/**
 * @file codegen.h
 * @brief Code generation macros for mathematical operations.
 */

#include "maths/promotion.h"
#include "maths/result.h"
#include "maths/validate.h"

/* Code Generation Macro Helpers */
#define ORT_MATH_FOREACH_INTEGER_TYPE(macro)              \
    macro(int8_t,   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)   \
    macro(int16_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)  \
    macro(int32_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)  \
    macro(int64_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)  \
    macro(uint8_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)  \
    macro(uint16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) \
    macro(uint32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)

#define ORT_MATH_FOREACH_REAL_TYPE(macro)                 \
    macro(float,   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)   \
    macro(double,  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)

#define ORT_MATH_FOREACH_NUMERIC_TYPE(macro)             \
    ORT_MATH_FOREACH_REAL_TYPE(macro)                    \
    ORT_MATH_FOREACH_INTEGER_TYPE(macro)

#define ORT_MATH_FOREACH_INTEGER_TYPE_EX(macro, ex)           \
    macro(int8_t,   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,   ex) \
    macro(int16_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,  ex) \
    macro(int32_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,  ex) \
    macro(int64_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,  ex) \
    macro(uint8_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,  ex) \
    macro(uint16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, ex) \
    macro(uint32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, ex)

#define ORT_MATH_FOREACH_REAL_TYPE_EX(macro, ex)              \
    macro(float,   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,   ex) \
    macro(double,  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,  ex)

#define ORT_MATH_FOREACH_NUMERIC_TYPE_EX(macro, ex)           \
    ORT_MATH_FOREACH_REAL_TYPE_EX(macro, ex)                  \
    ORT_MATH_FOREACH_INTEGER_TYPE_EX(macro, ex)

#define ORT_MATH_FOREACH_ALL_TYPES(macro)                     \
    ORT_MATH_FOREACH_NUMERIC_TYPE(macro)                      \
    macro(zend_bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)

#define ORT_MATH_FOREACH_ALL_TYPES_EX(macro, ex)              \
    ORT_MATH_FOREACH_NUMERIC_TYPE_EX(macro, ex)               \
    macro(zend_bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, ex)

#define ORT_MATH_BINARY_OP_IMPL(op_name, c_type, onnx_type, operator) \
void ort_math_frontend_##op_name##_##c_type(void* result,                  \
        const void* a, const void* b, size_t count) {                 \
    c_type* res = (c_type*)result;                                    \
    const c_type* va = (const c_type*)a;                              \
    const c_type* vb = (const c_type*)b;                              \
    for (size_t i = 0; i < count; i++) {                              \
        res[i] = va[i] operator vb[i];                                \
    }                                                                 \
}

#define ORT_MATH_BINARY_OP_EXTERN(op_name, c_type, onnx_type)     \
    extern void ort_math_frontend_##op_name##_##c_type(                \
        void* result, const void* a, const void* b, size_t count);

#define ORT_MATH_SCALAR_OP_IMPL(op_name, c_type, onnx_type, operator) \
void ort_math_frontend_##op_name##_scalar_##c_type(void* result,           \
        const void* a, const void* b, size_t count) {                 \
    c_type* res = (c_type*)result;                                    \
    const c_type* va = (const c_type*)a;                              \
    c_type sb = *(const c_type*)b;                                    \
    for (size_t i = 0; i < count; i++) {                              \
        res[i] = va[i] operator sb;                                   \
    }                                                                 \
}

#define ORT_MATH_SCALAR_OP_EXTERN(op_name, c_type, onnx_type)     \
    extern void ort_math_frontend_##op_name##_scalar_##c_type(         \
        void* result, const void* a, const void* b, size_t count);

/** {{{ Result Implementation Generation }}} */
#define ORT_MATH_BINARY_RESULT_IMPL(func_name, getter_func)             \
ort_tensor_t* ort_math_result_##func_name(                              \
        ort_tensor_t* tensor_a, ort_tensor_t* tensor_b) {               \
    if (!ort_math_validate_input(tensor_a, #func_name) ||               \
        !ort_math_validate_input(tensor_b, #func_name)) {               \
        return NULL;                                                    \
    }                                                                   \
                                                                        \
    ort_math_type_promotion_t promotion =                               \
        ort_math_type_promote(                                          \
            tensor_a->type, tensor_b->type);                            \
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
        tensor_a, tensor_b, op_func, #func_name);                       \
}

#define ORT_MATH_SCALAR_RESULT_IMPL(func_name, getter_func)             \
ort_tensor_t* ort_math_result_##func_name##_scalar(                     \
    ort_tensor_t* tensor, zval* scalar) {                               \
    if (!ort_math_validate_input(tensor, #func_name)) {                 \
        return NULL;                                                    \
    }                                                                   \
                                                                        \
    ort_math_scalar_op_func_t op_func = getter_func(tensor->type);      \
    if (!op_func) {                                                     \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,        \
            #func_name ": unsupported data type for tensor operation"); \
        return NULL;                                                    \
    }                                                                   \
                                                                        \
    return ort_math_result_element_wise_scalar(                         \
        tensor, scalar, op_func, #func_name);                           \
} 

#define ORT_MATH_UNARY_RESULT_IMPL(func_name, getter_func)                   \
ort_tensor_t* ort_math_result_##func_name(                                   \
    ort_tensor_t* tensor) {                                                  \
    if (!ort_math_validate_input(tensor, #func_name)) {                      \
        return NULL;                                                         \
    }                                                                        \
    ort_math_unary_op_func_t op_func = getter_func(tensor->type);            \
    if (!op_func) {                                                          \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,             \
            #func_name ": unsupported data type for mathematical function"); \
        return NULL;                                                         \
    }                                                                        \
    return ort_math_result_element_wise_unary(tensor, op_func, #func_name);  \
}

/* {{{ Switch Casing for Getter Functions */
#define ORT_MATH_BINARY_FUNC_GETTER_CASE(c_type, onnx_type, op_name) \
    case onnx_type: return ort_math_frontend_##op_name##_##c_type;

#define ORT_MATH_SCALAR_FUNC_GETTER_CASE(c_type, onnx_type, op_name) \
    case onnx_type: return ort_math_frontend_##op_name##_scalar_##c_type; /* }}} */

#endif
