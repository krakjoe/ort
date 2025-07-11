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

/* {{{ Type Iteration Macro Definitions */
#define ORT_MATH_FOREACH_SIGNED_TYPE(macro)               \
    macro(int8_t,   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)   \
    macro(int16_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)  \
    macro(int32_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)  \
    macro(int64_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)

#define ORT_MATH_FOREACH_UNSIGNED_TYPE(macro)             \
    macro(uint8_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)  \
    macro(uint16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) \
    macro(uint32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)

#define ORT_MATH_FOREACH_INTEGER_TYPE(macro)              \
    ORT_MATH_FOREACH_SIGNED_TYPE(macro)                   \
    ORT_MATH_FOREACH_UNSIGNED_TYPE(macro)

#define ORT_MATH_FOREACH_REAL_TYPE(macro)                 \
    macro(float,   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)   \
    macro(double,  ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)

#define ORT_MATH_FOREACH_NUMERIC_TYPE(macro)              \
    ORT_MATH_FOREACH_REAL_TYPE(macro)                     \
    ORT_MATH_FOREACH_INTEGER_TYPE(macro)

#define ORT_MATH_FOREACH_ALL_TYPES(macro)                 \
    ORT_MATH_FOREACH_NUMERIC_TYPE(macro)                  \
    macro(zend_bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) /* }}} */

/* {{{ Frontend Function Declaration Macros **/
#define ORT_MATH_FRONTEND_OP_SYMBOL(op, type) \
    ort_math_frontend_##op##_##type
#define ORT_MATH_FRONTEND_OP_AXIS_SYMBOL(op, type) \
    ort_math_frontend_##op##_axis_##type
#define ORT_MATH_FRONTEND_BINARY_OP_DECL(op, type)                    \
    void ORT_MATH_FRONTEND_OP_SYMBOL(op, type)(                       \
        void* result, const void* a, const void* b, size_t count)
#define ORT_MATH_FRONTEND_UNARY_OP_DECL(op, type)                     \
    void ORT_MATH_FRONTEND_OP_SYMBOL(op, type)(                       \
        void* result, const void* a, size_t count)
#define ORT_MATH_FRONTEND_MATMUL_OP_DECL(type)                        \
    void ORT_MATH_FRONTEND_OP_SYMBOL(matmul, type)(                   \
        void* result, const void* a, const void* b,                   \
        size_t a_rows, size_t a_cols, size_t b_cols)
#define ORT_MATH_FRONTEND_REDUCTION_OP_DECL ORT_MATH_FRONTEND_UNARY_OP_DECL
#define ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(op, type)            \
    void ORT_MATH_FRONTEND_OP_AXIS_SYMBOL(op, type) (                 \
        void* result, const void* a,                                  \
        size_t outer, size_t axis, size_t inner) 
#define ORT_MATH_FRONTEND_SERIAL_REDUCTION_AXIS_OP_DECL(op, type)            \
    void ORT_MATH_FRONTEND_OP_AXIS_SYMBOL(serial_##op, type) (                 \
        void* result, const void* a,                                  \
        size_t outer, size_t axis, size_t inner) /* }}} */

/* {{{ Backend Function Declaration Macros **/
#define ORT_MATH_BACKEND_OP_SYMBOL(op, type) \
    ort_math_backend_##op##_##type
#define ORT_MATH_BACKEND_OP_AXIS_SYMBOL(op, type) \
    ort_math_backend_##op##_axis_##type
#define ORT_MATH_BACKEND_BINARY_OP_DECL(op, type)                  \
    void ORT_MATH_BACKEND_OP_SYMBOL(op, type) (                    \
        void* result, const void* a, const void* b, size_t count)
#define ORT_MATH_BACKEND_UNARY_OP_DECL(op, type)                   \
    void ORT_MATH_BACKEND_OP_SYMBOL(op, type) (                    \
        void* result, const void* a, size_t count)
#define ORT_MATH_BACKEND_MATMUL_OP_DECL(type)                      \
    void ORT_MATH_BACKEND_OP_SYMBOL(matmul, type) (                \
        void* result, const void* a, const void* b,                \
        size_t a_rows, size_t a_cols, size_t b_cols)
#define ORT_MATH_BACKEND_REDUCTION_OP_DECL ORT_MATH_BACKEND_UNARY_OP_DECL
#define ORT_MATH_BACKEND_REDUCTION_AXIS_OP_DECL(op, type)          \
    void ORT_MATH_BACKEND_OP_AXIS_SYMBOL(op, type) (               \
        void* result, const void* a,                               \
        size_t outer, size_t axis, size_t inner) /* }}} */
#define ORT_MATH_BACKEND_SERIAL_REDUCTION_AXIS_OP_DECL(op, type)   \
    void ORT_MATH_BACKEND_OP_AXIS_SYMBOL(serial_##op, type) (      \
        void* result, const void* a,                               \
        size_t outer, size_t axis, size_t inner) /* }}} */
#define ORT_MATH_BINARY_OP_IMPL(op_name, c_type, onnx_type, operator) \
void ort_math_frontend_##op_name##_##c_type(void* result,             \
        const void* a, const void* b, size_t count) {                 \
    c_type* res = (c_type*)result;                                    \
    const c_type* va = (const c_type*)a;                              \
    const c_type* vb = (const c_type*)b;                              \
    for (size_t i = 0; i < count; i++) {                              \
        res[i] = va[i] operator vb[i];                                \
    }                                                                 \
}

#define ORT_MATH_BINARY_OP_EXTERN(op_name, c_type, onnx_type)         \
    extern void ort_math_frontend_##op_name##_##c_type(               \
        void* result, const void* a, const void* b, size_t count);

#define ORT_MATH_SCALAR_OP_IMPL(op_name, c_type, onnx_type, operator) \
void ort_math_frontend_##op_name##_scalar_##c_type(void* result,      \
        const void* a, const void* b, size_t count) {                 \
    c_type* res = (c_type*)result;                                    \
    const c_type* va = (const c_type*)a;                              \
    c_type sb = *(const c_type*)b;                                    \
    for (size_t i = 0; i < count; i++) {                              \
        res[i] = va[i] operator sb;                                   \
    }                                                                 \
}

#define ORT_MATH_SCALAR_OP_EXTERN(op_name, c_type, onnx_type)         \
    extern void ort_math_frontend_##op_name##_scalar_##c_type(        \
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
        ort_math_type_promote(tensor_a, tensor_b);                      \
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

#define ORT_MATH_SERIAL_REDUCE_TENSOR_RESULT_IMPL(func_name, getter_func, validate_tensor_func) \
ort_tensor_t* ort_math_result_serial_reduce_tensor_##func_name(                     \
    ort_tensor_t* tensor) {                                                  \
    if (!validate_tensor_func(tensor, #func_name)) {                         \
        return NULL;                                                         \
    }                                                                        \
    ort_math_unary_op_func_t op_func = getter_func(tensor->type);            \
    if (!op_func) {                                                          \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,             \
            #func_name ": unsupported data type for mathematical function"); \
        return NULL;                                                         \
    }                                                                        \
    return ort_math_result_serial_element_wise_reduce_tensor(                       \
        NULL, tensor, op_func, #func_name);                                  \
}

#define ORT_MATH_REDUCE_TENSOR_RESULT_IMPL(func_name, getter_func, validate_tensor_func) \
ort_tensor_t* ort_math_result_reduce_tensor_##func_name(                     \
    ort_tensor_t* tensor) {                                                  \
    if (!validate_tensor_func(tensor, #func_name)) {                         \
        return NULL;                                                         \
    }                                                                        \
    ort_math_unary_op_func_t op_func = getter_func(tensor->type);            \
    if (!op_func) {                                                          \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,             \
            #func_name ": unsupported data type for mathematical function"); \
        return NULL;                                                         \
    }                                                                        \
    return ort_math_result_element_wise_reduce_tensor(                       \
        NULL, tensor, op_func, #func_name);                                  \
}

#define ORT_MATH_SERIAL_REDUCE_TENSOR_PROMOTE_RESULT_IMPL(\
    func_name, getter_func, validate_tensor_func, promotion_func) \
ort_tensor_t* ort_math_result_reduce_tensor_##func_name(                     \
    ort_tensor_t* tensor) {                                                  \
    if (!validate_tensor_func(tensor, #func_name)) {                         \
        return NULL;                                                         \
    }                                                                        \
    ort_math_type_promotion_t promotion =                                    \
        ort_math_operation_promote(                                          \
            promotion_func(tensor->type), 1, tensor);                        \
    ort_math_unary_op_func_t op_func = getter_func(promotion.result_type);   \
    if (!op_func) {                                                          \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,             \
            #func_name ": unsupported data type for mathematical function"); \
        return NULL;                                                         \
    }                                                                        \
    return ort_math_result_serial_element_wise_reduce_tensor(                \
        &promotion, tensor, op_func, #func_name);                            \
}

#define ORT_MATH_REDUCE_TENSOR_PROMOTE_RESULT_IMPL(\
    func_name, getter_func, validate_tensor_func, promotion_func) \
ort_tensor_t* ort_math_result_reduce_tensor_##func_name(                     \
    ort_tensor_t* tensor) {                                                  \
    if (!validate_tensor_func(tensor, #func_name)) {                         \
        return NULL;                                                         \
    }                                                                        \
    ort_math_type_promotion_t promotion =                                    \
        ort_math_operation_promote(                                          \
            promotion_func(tensor->type), 1, tensor);                                      \
    ort_math_unary_op_func_t op_func = getter_func(promotion.result_type);   \
    if (!op_func) {                                                          \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,             \
            #func_name ": unsupported data type for mathematical function"); \
        return NULL;                                                         \
    }                                                                        \
    return ort_math_result_element_wise_reduce_tensor(                       \
        &promotion, tensor, op_func, #func_name);                            \
}

#define ORT_MATH_SERIAL_REDUCE_AXIS_RESULT_IMPL(                                      \
    func_name, getter_func, validate_tensor_func,validate_axis_func, shape_func) \
ort_tensor_t* ort_math_result_serial_reduce_axis_##func_name(                         \
    ort_tensor_t* tensor, zend_long axis, zend_bool keepdims) {                \
    if (!validate_tensor_func(tensor, #func_name)) {                           \
        return NULL;                                                           \
    }                                                                          \
    if (!validate_axis_func(tensor, &axis, #func_name)) {                      \
        return NULL;                                                           \
    }                                                                          \
    ort_math_reduction_op_func_t op_func = getter_func(tensor->type);          \
    if (!op_func) {                                                            \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,               \
            #func_name ": unsupported data type for mathematical function");   \
        return NULL;                                                           \
    }                                                                          \
    return ort_math_result_serial_element_wise_reduce_axis(NULL,                      \
        tensor, axis, keepdims,                                                \
        op_func, #func_name, shape_func);                                      \
}

#define ORT_MATH_REDUCE_AXIS_RESULT_IMPL(                                      \
    func_name, getter_func, validate_tensor_func,validate_axis_func, shape_func) \
ort_tensor_t* ort_math_result_reduce_axis_##func_name(                         \
    ort_tensor_t* tensor, zend_long axis, zend_bool keepdims) {                \
    if (!validate_tensor_func(tensor, #func_name)) {                           \
        return NULL;                                                           \
    }                                                                          \
    if (!validate_axis_func(tensor, &axis, #func_name)) {                      \
        return NULL;                                                           \
    }                                                                          \
    ort_math_reduction_op_func_t op_func = getter_func(tensor->type);          \
    if (!op_func) {                                                            \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,               \
            #func_name ": unsupported data type for mathematical function");   \
        return NULL;                                                           \
    }                                                                          \
    return ort_math_result_element_wise_reduce_axis(NULL,                      \
        tensor, axis, keepdims,                                                \
        op_func, #func_name, shape_func);                                      \
}

#define ORT_MATH_SERIAL_REDUCE_AXIS_PROMOTE_RESULT_IMPL(                              \
    func_name, getter_func,                                                    \
    validate_tensor_func, validate_axis_func,                                  \
    shape_func, promotion_func)                                                \
ort_tensor_t* ort_math_result_reduce_axis_##func_name(                         \
    ort_tensor_t* tensor, zend_long axis, zend_bool keepdims) {                \
    if (!validate_tensor_func(tensor, #func_name)) {                           \
        return NULL;                                                           \
    }                                                                          \
    if (!validate_axis_func(tensor, &axis, #func_name)) {                      \
        return NULL;                                                           \
    }                                                                          \
    ort_math_type_promotion_t promotion =                                      \
        ort_math_operation_promote(                                            \
            promotion_func(tensor->type), 1, tensor);                          \
    ort_math_reduction_op_func_t op_func = getter_func(promotion.result_type); \
    if (!op_func) {                                                            \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,               \
            #func_name ": unsupported data type for mathematical function");   \
        return NULL;                                                           \
    }                                                                          \
    return ort_math_result_serial_element_wise_reduce_axis(&promotion,                \
        tensor, axis, keepdims,                                                \
        op_func, #func_name, shape_func);                                      \
}

#define ORT_MATH_REDUCE_AXIS_PROMOTE_RESULT_IMPL(                              \
    func_name, getter_func,                                                    \
    validate_tensor_func, validate_axis_func,                                  \
    shape_func, promotion_func)                                                \
ort_tensor_t* ort_math_result_reduce_axis_##func_name(                         \
    ort_tensor_t* tensor, zend_long axis, zend_bool keepdims) {                \
    if (!validate_tensor_func(tensor, #func_name)) {                           \
        return NULL;                                                           \
    }                                                                          \
    if (!validate_axis_func(tensor, &axis, #func_name)) {                      \
        return NULL;                                                           \
    }                                                                          \
    ort_math_type_promotion_t promotion =                                      \
        ort_math_operation_promote(                                            \
            promotion_func(tensor->type), 1, tensor);                          \
    ort_math_reduction_op_func_t op_func = getter_func(promotion.result_type); \
    if (!op_func) {                                                            \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,               \
            #func_name ": unsupported data type for mathematical function");   \
        return NULL;                                                           \
    }                                                                          \
    return ort_math_result_element_wise_reduce_axis(&promotion,                \
        tensor, axis, keepdims,                                                \
        op_func, #func_name, shape_func);                                      \
}


/* {{{ Switch Casing for Getter Functions */
#define ORT_MATH_BINARY_FUNC_GETTER_CASE(c_type, onnx_type, op_name) \
    case onnx_type: return ORT_MATH_FRONTEND_OP_SYMBOL(op_name, c_type);

#define ORT_MATH_SCALAR_FUNC_GETTER_CASE(c_type, onnx_type, op_name) \
    case onnx_type: return ORT_MATH_FRONTEND_OP_SYMBOL(op_name, scalar_##c_type); /* }}} */

#endif
