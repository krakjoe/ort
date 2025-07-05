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

#include <php.h>

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
void ort_math_ops_##op_name##_##c_type(void* result,                  \
        const void* a, const void* b, size_t count) {                 \
    c_type* res = (c_type*)result;                                    \
    const c_type* va = (const c_type*)a;                              \
    const c_type* vb = (const c_type*)b;                              \
    for (size_t i = 0; i < count; i++) {                              \
        res[i] = va[i] operator vb[i];                                \
    }                                                                 \
}

#define ORT_MATH_SCALAR_OP_IMPL(op_name, c_type, onnx_type, operator) \
void ort_math_ops_##op_name##_scalar_##c_type(void* result,           \
        const void* a, const void* b, size_t count) {                 \
    c_type* res = (c_type*)result;                                    \
    const c_type* va = (const c_type*)a;                              \
    c_type sb = *(const c_type*)b;                                    \
    for (size_t i = 0; i < count; i++) {                              \
        res[i] = va[i] operator sb;                                   \
    }                                                                 \
}

/** {{{ Result Implementation Generation }}} */
#define ORT_MATH_BINARY_RESULT_IMPL(func_name, getter_func)             \
ort_math_result_t* ort_math_result_##func_name(                         \
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
ort_math_result_t* ort_math_result_##func_name##_scalar(                \
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
ort_math_result_t* ort_math_result_##func_name(                              \
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
}/* }}} */

/* {{{ Switch Casing for Getter Functions */
#define ORT_MATH_BINARY_FUNC_GETTER_CASE(c_type, onnx_type, op_name) \
    case onnx_type: return ort_math_ops_##op_name##_##c_type;

#define ORT_MATH_SCALAR_FUNC_GETTER_CASE(c_type, onnx_type, op_name) \
    case onnx_type: return ort_math_ops_##op_name##_scalar_##c_type; /* }}} */

/* Mathematical operation implementations */
extern void ort_math_ops_add_float(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_double(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_int8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_int16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_int32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_int64_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_uint8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_uint16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_uint32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_zend_bool(void* result, const void* a, const void* b, size_t count);

extern void ort_math_ops_sub_float(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_double(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_int8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_int16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_int32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_int64_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_uint8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_uint16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_uint32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_zend_bool(void* result, const void* a, const void* b, size_t count);

extern void ort_math_ops_mul_float(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_double(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_int8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_int16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_int32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_int64_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_uint8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_uint16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_uint32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_zend_bool(void* result, const void* a, const void* b, size_t count);

extern void ort_math_ops_div_float(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_double(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_int8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_int16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_int32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_int64_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_uint8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_uint16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_uint32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_zend_bool(void* result, const void* a, const void* b, size_t count);

extern void ort_math_ops_sqrt_float(void* result, const void* a, size_t count);
extern void ort_math_ops_sqrt_double(void* result, const void* a, size_t count);

extern void ort_math_ops_ceil_float(void* result, const void* a, size_t count);
extern void ort_math_ops_ceil_double(void* result, const void* a, size_t count);

extern void ort_math_ops_floor_float(void* result, const void* a, size_t count);
extern void ort_math_ops_floor_double(void* result, const void* a, size_t count);

extern void ort_math_ops_round_float(void* result, const void* a, size_t count);
extern void ort_math_ops_round_double(void* result, const void* a, size_t count);

extern void ort_math_ops_abs_float(void* result, const void* a, size_t count);
extern void ort_math_ops_abs_double(void* result, const void* a, size_t count);

extern void ort_math_ops_sign_float(void* result, const void* a, size_t count);
extern void ort_math_ops_sign_double(void* result, const void* a, size_t count);

extern void ort_math_ops_reciprocal_float(void* result, const void* a, size_t count);
extern void ort_math_ops_reciprocal_double(void* result, const void* a, size_t count);

extern void ort_math_ops_negative_float(void* result, const void* a, size_t count);
extern void ort_math_ops_negative_double(void* result, const void* a, size_t count);

extern void ort_math_ops_mod_float(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_double(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_int8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_int16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_int32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_int64_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_uint8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_uint16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_uint32_t(void* result, const void* a, const void* b, size_t count);

extern void ort_math_ops_pow_float(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_pow_double(void* result, const void* a, const void* b, size_t count);

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

    /* SIMD-optimized binary operations (NULL if not available) */
    ort_math_element_op_func_t add_simd_func;
    ort_math_element_op_func_t sub_simd_func;
    ort_math_element_op_func_t mul_simd_func;
    ort_math_element_op_func_t div_simd_func;
    
    /* Comparison operations */
    ort_math_element_op_func_t eq_func;
    ort_math_element_op_func_t ne_func;
    ort_math_element_op_func_t gt_func;
    ort_math_element_op_func_t ge_func;
    ort_math_element_op_func_t lt_func;
    ort_math_element_op_func_t le_func;

    /* Scalar operations */
    ort_math_scalar_op_func_t add_scalar_func;
    ort_math_scalar_op_func_t sub_scalar_func;
    ort_math_scalar_op_func_t mul_scalar_func;
    ort_math_scalar_op_func_t div_scalar_func;
    ort_math_scalar_op_func_t pow_scalar_func;
    ort_math_scalar_op_func_t mod_scalar_func;

    /* Unary operations */
    ort_math_unary_op_func_t negative_func;
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
    ort_math_unary_op_func_t abs_func;
    ort_math_unary_op_func_t sign_func;
    ort_math_unary_op_func_t reciprocal_func;
    
    /* SIMD-optimized unary operations (NULL if not available) */
    ort_math_unary_op_func_t sqrt_simd_func;
    ort_math_unary_op_func_t negative_simd_func;
    ort_math_unary_op_func_t ceil_simd_func;
    ort_math_unary_op_func_t floor_simd_func;
    ort_math_unary_op_func_t round_simd_func;
    ort_math_unary_op_func_t abs_simd_func;
    ort_math_unary_op_func_t sign_simd_func;
    ort_math_unary_op_func_t reciprocal_simd_func;
} ort_math_type_dispatch_t;

/* Function dispatch table access */
const ort_math_type_dispatch_t* ort_math_get_dispatch(ONNXTensorElementDataType type);

/* Memory management functions */
void ort_math_broadcast_info_destroy(ort_math_broadcast_info_t* info);
void ort_math_result_destroy(ort_math_result_t* result);
#endif
