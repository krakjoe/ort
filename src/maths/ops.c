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

 #include <math.h>

 #include "status.h"

#include "maths/ops.h"
#include "maths/core.h"

/* Binary element-wise operation implementation */
#define ORT_MATH_BINARY_OP_IMPL(op_name, c_type, onnx_type, operator) \
void ort_math_ops_##op_name##_##c_type(void* result, const void* a, const void* b, size_t count) { \
    c_type* res = (c_type*)result; \
    const c_type* va = (const c_type*)a; \
    const c_type* vb = (const c_type*)b; \
    for (size_t i = 0; i < count; i++) { \
        res[i] = va[i] operator vb[i]; \
    } \
}

/* Boolean-specific binary operations using logical operators */
#define ORT_MATH_BINARY_BOOL_OP_IMPL(op_name, operator) \
void ort_math_ops_##op_name##_zend_bool(void* result, const void* a, const void* b, size_t count) { \
    zend_bool* res = (zend_bool*)result; \
    const zend_bool* va = (const zend_bool*)a; \
    const zend_bool* vb = (const zend_bool*)b; \
    for (size_t i = 0; i < count; i++) { \
        res[i] = va[i] operator vb[i]; \
    } \
}

/* Generate boolean operations */
ORT_MATH_BINARY_BOOL_OP_IMPL(add, ||)
ORT_MATH_BINARY_BOOL_OP_IMPL(sub, && !)
ORT_MATH_BINARY_BOOL_OP_IMPL(mul, &&)
ORT_MATH_BINARY_BOOL_OP_IMPL(div, &&)

/* Binary element-wise operation */
#define ORT_MATH_BINARY_DIV_IMPL(op_name, c_type, onnx_type) \
void ort_math_ops_##op_name##_##c_type(void* result, const void* a, const void* b, size_t count) { \
    c_type* res = (c_type*)result; \
    const c_type* va = (const c_type*)a; \
    const c_type* vb = (const c_type*)b; \
    for (size_t i = 0; i < count; i++) { \
        res[i] = va[i] / vb[i]; /* Let IEEE 754 handle div by zero */ \
    } \
}

/* Scalar operation implementation */
#define ORT_MATH_SCALAR_OP_IMPL(op_name, c_type, onnx_type, operator) \
void ort_math_ops_##op_name##_scalar_##c_type(void* result, const void* tensor_data, const void* scalar, size_t count) { \
    c_type* res = (c_type*)result; \
    const c_type* data = (const c_type*)tensor_data; \
    c_type scalar_val = *(const c_type*)scalar; \
    for (size_t i = 0; i < count; i++) { \
        res[i] = data[i] operator scalar_val; \
    } \
}

/* Boolean-specific scalar operations */
#define ORT_MATH_SCALAR_BOOL_OP_IMPL(op_name, logical_op) \
void ort_math_ops_##op_name##_scalar_zend_bool(void* result, const void* tensor_data, const void* scalar, size_t count) { \
    zend_bool* res = (zend_bool*)result; \
    const zend_bool* data = (const zend_bool*)tensor_data; \
    zend_bool scalar_val = *(const zend_bool*)scalar; \
    for (size_t i = 0; i < count; i++) { \
        res[i] = logical_op; \
    } \
}

/* Generate boolean scalar operations */
ORT_MATH_SCALAR_BOOL_OP_IMPL(add, data[i] || scalar_val)
ORT_MATH_SCALAR_BOOL_OP_IMPL(sub, data[i] && !scalar_val)
ORT_MATH_SCALAR_BOOL_OP_IMPL(mul, data[i] && scalar_val)
ORT_MATH_SCALAR_BOOL_OP_IMPL(div, data[i] && scalar_val)

/* Scalar division (follows IEEE 754 standards for division by zero) */
#define ORT_MATH_SCALAR_DIV_IMPL(op_name, c_type, onnx_type) \
void ort_math_ops_##op_name##_scalar_##c_type(void* result, const void* tensor_data, const void* scalar, size_t count) { \
    c_type* res = (c_type*)result; \
    const c_type* data = (const c_type*)tensor_data; \
    c_type scalar_val = *(const c_type*)scalar; \
    for (size_t i = 0; i < count; i++) { \
        res[i] = data[i] / scalar_val; /* Let IEEE 754 handle div by zero */ \
    } \
}

/* Function getter macro */
#define ORT_MATH_FUNC_GETTER_CASE(c_type, onnx_type, op_name) \
    case onnx_type: return ort_math_ops_##op_name##_##c_type;

/* =============================================================================
 * ELEMENT-WISE BINARY OPERATIONS
 * =============================================================================
 */

/* Addition operations */
#define ORT_MATH_ADD_IMPL(c_type, onnx_type) ORT_MATH_BINARY_OP_IMPL(add, c_type, onnx_type, +)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_ADD_IMPL)

static ort_math_element_op_func_t ort_math_ops_get_add_func(ONNXTensorElementDataType type) {
    switch (type) {
#define ORT_MATH_ADD_CASE(c_type, onnx_type) ORT_MATH_FUNC_GETTER_CASE(c_type, onnx_type, add)
        ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_ADD_CASE)
#undef ORT_MATH_ADD_CASE
        default: return NULL;
    }
}

/* Subtraction operations */
#define ORT_MATH_SUB_IMPL(c_type, onnx_type) ORT_MATH_BINARY_OP_IMPL(sub, c_type, onnx_type, -)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_SUB_IMPL)

static ort_math_element_op_func_t ort_math_ops_get_sub_func(ONNXTensorElementDataType type) {
    switch (type) {
#define ORT_MATH_SUB_CASE(c_type, onnx_type) ORT_MATH_FUNC_GETTER_CASE(c_type, onnx_type, sub)
        ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_SUB_CASE)
#undef ORT_MATH_SUB_CASE
        default: return NULL;
    }
}

/* Multiplication operations */
#define ORT_MATH_MUL_IMPL(c_type, onnx_type) ORT_MATH_BINARY_OP_IMPL(mul, c_type, onnx_type, *)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_MUL_IMPL)

static ort_math_element_op_func_t ort_math_ops_get_mul_func(ONNXTensorElementDataType type) {
    switch (type) {
#define ORT_MATH_MUL_CASE(c_type, onnx_type) ORT_MATH_FUNC_GETTER_CASE(c_type, onnx_type, mul)
        ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_MUL_CASE)
#undef ORT_MATH_MUL_CASE
        default: return NULL;
    }
}

/* Division operations */
#define ORT_MATH_DIV_IMPL(c_type, onnx_type) ORT_MATH_BINARY_DIV_IMPL(div, c_type, onnx_type)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_DIV_IMPL)

static ort_math_element_op_func_t ort_math_ops_get_div_func(ONNXTensorElementDataType type) {
    switch (type) {
#define ORT_MATH_DIV_CASE(c_type, onnx_type) ORT_MATH_FUNC_GETTER_CASE(c_type, onnx_type, div)
        ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_DIV_CASE)
#undef ORT_MATH_DIV_CASE
        default: return NULL;
    }
}

/* =============================================================================
 * HIGH-LEVEL OPERATION IMPLEMENTATIONS
 * =============================================================================
 */

#define ORT_MATH_BINARY_RESULT_IMPL(func_name, getter_func) \
ort_math_result_t* ort_math_result_##func_name(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b) { \
    if (!ort_math_validate_input(tensor_a, #func_name) || !ort_math_validate_input(tensor_b, #func_name)) { \
        return NULL; \
    } \
    \
    ort_math_type_promotion_t promotion = ort_math_type_promote(tensor_a->type, tensor_b->type); \
    if (!promotion.is_valid) { \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce, \
            #func_name ": incompatible types for operation"); \
        return NULL; \
    } \
    \
    ort_math_element_op_func_t op_func = getter_func(promotion.result_type); \
    if (!op_func) { \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce, \
            #func_name ": unsupported data type for tensor operation"); \
        return NULL; \
    } \
    \
    return ort_math_result_element_wise_binary(tensor_a, tensor_b, op_func, #func_name); \
}

ORT_MATH_BINARY_RESULT_IMPL(add,      ort_math_ops_get_add_func);
ORT_MATH_BINARY_RESULT_IMPL(subtract, ort_math_ops_get_sub_func);
ORT_MATH_BINARY_RESULT_IMPL(multiply, ort_math_ops_get_mul_func);
ORT_MATH_BINARY_RESULT_IMPL(divide,   ort_math_ops_get_div_func);

/* =============================================================================
 * SCALAR OPERATIONS
 * =============================================================================
 */

/* Scalar addition */
#define ORT_MATH_ADD_SCALAR_IMPL(c_type, onnx_type) ORT_MATH_SCALAR_OP_IMPL(add, c_type, onnx_type, +)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_ADD_SCALAR_IMPL)

static ort_math_scalar_op_func_t ort_math_ops_get_add_scalar_func(ONNXTensorElementDataType type) {
    switch (type) {
#define ORT_MATH_ADD_SCALAR_CASE(c_type, onnx_type) case onnx_type: return ort_math_ops_add_scalar_##c_type;
        ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_ADD_SCALAR_CASE)
#undef ORT_MATH_ADD_SCALAR_CASE
        default: return NULL;
    }
}

/* Scalar subtraction */
#define ORT_MATH_SUB_SCALAR_IMPL(c_type, onnx_type) ORT_MATH_SCALAR_OP_IMPL(sub, c_type, onnx_type, -)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_SUB_SCALAR_IMPL)

static ort_math_scalar_op_func_t ort_math_ops_get_sub_scalar_func(ONNXTensorElementDataType type) {
    switch (type) {
#define ORT_MATH_SUB_SCALAR_CASE(c_type, onnx_type) case onnx_type: return ort_math_ops_sub_scalar_##c_type;
        ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_SUB_SCALAR_CASE)
#undef ORT_MATH_SUB_SCALAR_CASE
        default: return NULL;
    }
}

/* Scalar multiplication */
#define ORT_MATH_MUL_SCALAR_IMPL(c_type, onnx_type) ORT_MATH_SCALAR_OP_IMPL(mul, c_type, onnx_type, *)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_MUL_SCALAR_IMPL)

static ort_math_scalar_op_func_t ort_math_ops_get_mul_scalar_func(ONNXTensorElementDataType type) {
    switch (type) {
#define ORT_MATH_MUL_SCALAR_CASE(c_type, onnx_type) case onnx_type: return ort_math_ops_mul_scalar_##c_type;
        ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_MUL_SCALAR_CASE)
#undef ORT_MATH_MUL_SCALAR_CASE
        default: return NULL;
    }
}

/* Scalar division */
#define ORT_MATH_DIV_SCALAR_IMPL(c_type, onnx_type) ORT_MATH_SCALAR_DIV_IMPL(div, c_type, onnx_type)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_DIV_SCALAR_IMPL)

static ort_math_scalar_op_func_t ort_math_ops_get_div_scalar_func(ONNXTensorElementDataType type) {
    switch (type) {
#define ORT_MATH_DIV_SCALAR_CASE(c_type, onnx_type) case onnx_type: return ort_math_ops_div_scalar_##c_type;
        ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_DIV_SCALAR_CASE)
#undef ORT_MATH_DIV_SCALAR_CASE
        default: return NULL;
    }
}

#define ORT_MATH_SCALAR_RESULT_IMPL(func_name, getter_func) \
ort_math_result_t* ort_math_result_##func_name##_scalar(ort_tensor_t* tensor, zval* scalar) { \
    if (!ort_math_validate_input(tensor, #func_name)) { \
        return NULL; \
    } \
    \
    ort_math_scalar_op_func_t op_func = getter_func(tensor->type); \
    if (!op_func) { \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce, \
            #func_name ": unsupported data type for tensor operation"); \
        return NULL; \
    } \
    \
    return ort_math_result_element_wise_scalar(tensor, scalar, op_func, #func_name); \
}

ORT_MATH_SCALAR_RESULT_IMPL(add,      ort_math_ops_get_add_scalar_func);
ORT_MATH_SCALAR_RESULT_IMPL(subtract, ort_math_ops_get_sub_scalar_func);
ORT_MATH_SCALAR_RESULT_IMPL(multiply, ort_math_ops_get_mul_scalar_func);
ORT_MATH_SCALAR_RESULT_IMPL(divide,   ort_math_ops_get_div_scalar_func);

/* =============================================================================
 * POWER OPERATIONS
 * =============================================================================
 */

/* Power operations (floating point only) */
void ort_math_ops_pow_float(void* result, const void* a, const void* b, size_t count) {
    float* res = (float*)result;
    const float* base = (const float*)a;
    const float* exp = (const float*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = powf(base[i], exp[i]);
    }
}

void ort_math_ops_pow_double(void* result, const void* a, const void* b, size_t count) {
    double* res = (double*)result;
    const double* base = (const double*)a;
    const double* exp = (const double*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = pow(base[i], exp[i]);
    }
}

static ort_math_element_op_func_t ort_math_ops_get_pow_func(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return ort_math_ops_pow_float;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return ort_math_ops_pow_double;
        default: return NULL;
    }
}

/* Scalar power (floating point only) */
void ort_math_ops_pow_scalar_float(void* result, const void* tensor_data, const void* scalar, size_t count) {
    float* res = (float*)result;
    const float* data = (const float*)tensor_data;
    float scalar_val = *(const float*)scalar;
    for (size_t i = 0; i < count; i++) {
        res[i] = powf(data[i], scalar_val);
    }
}

void ort_math_ops_pow_scalar_double(void* result, const void* tensor_data, const void* scalar, size_t count) {
    double* res = (double*)result;
    const double* data = (const double*)tensor_data;
    double scalar_val = *(const double*)scalar;
    for (size_t i = 0; i < count; i++) {
        res[i] = pow(data[i], scalar_val);
    }
}

static ort_math_scalar_op_func_t ort_math_ops_get_pow_scalar_func(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return ort_math_ops_pow_scalar_float;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return ort_math_ops_pow_scalar_double;
        default: return NULL;
    }
}

ort_math_result_t* ort_math_result_power(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b) {
    if (!ort_math_validate_input(tensor_a, "power") || !ort_math_validate_input(tensor_b, "power")) {
        return NULL;
    }
    
    ort_math_element_op_func_t op_func = ort_math_ops_get_pow_func(tensor_a->type);
    if (!op_func) {
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,
            "power: unsupported data type for tensor operation");
        return NULL;
    }
    
    return ort_math_result_element_wise_binary(tensor_a, tensor_b, op_func, "power");
}

ort_math_result_t* ort_math_result_power_scalar(ort_tensor_t* tensor, zval* scalar) {
    if (!ort_math_validate_input(tensor, "power")) {
        return NULL;
    }
    
    ort_math_scalar_op_func_t op_func = ort_math_ops_get_pow_scalar_func(tensor->type);
    if (!op_func) {
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,
            "power: unsupported data type for scalar operation");
        return NULL;
    }
    
    return ort_math_result_element_wise_scalar(tensor, scalar, op_func, "power");
}

/* =============================================================================
 * UNARY MATHEMATICAL FUNCTIONS
 * =============================================================================
 */

/* Negation operation implementation */
#define ORT_MATH_NEGATIVE_IMPL(c_type, onnx_type) \
void ort_math_ops_negative_##c_type(void* result, const void* input, size_t count) { \
    c_type* res = (c_type*)result; \
    const c_type* data = (const c_type*)input; \
    for (size_t i = 0; i < count; i++) { \
        res[i] = -data[i]; \
    } \
}

/* Generate negation operation for all numeric types */
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_NEGATIVE_IMPL)

/* Boolean negation operation */
void ort_math_ops_negative_zend_bool(void* result, const void* input, size_t count) {
    zend_bool* res = (zend_bool*)result;
    const zend_bool* data = (const zend_bool*)input;
    for (size_t i = 0; i < count; i++) {
        res[i] = !data[i];
    }
}

static ort_math_unary_op_func_t ort_math_ops_get_negative_func(ONNXTensorElementDataType type) {
    /* Check for SIMD optimization first */
    const ort_math_type_dispatch_t* dispatch = ort_math_get_dispatch(type);
    if (dispatch && dispatch->negative_simd_func) {
        return dispatch->negative_simd_func;
    }
    
    /* Fall back to scalar implementation */
    switch (type) {
#define ORT_MATH_NEGATIVE_CASE(c_type, onnx_type) \
        case onnx_type: return ort_math_ops_negative_##c_type;
        ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_NEGATIVE_CASE)
#undef ORT_MATH_NEGATIVE_CASE
        default: return NULL;
    }
}

ort_math_result_t* ort_math_result_negative(ort_tensor_t* tensor) {
    if (!ort_math_validate_input(tensor, "negative")) {
        return NULL;
    }
    
    ort_math_unary_op_func_t op_func = ort_math_ops_get_negative_func(tensor->type);
    if (!op_func) {
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,
            "negative: unsupported data type for tensor operation");
        return NULL;
    }
    
    return ort_math_result_element_wise_unary(tensor, op_func, "negative");
}

#define ORT_MATH_SQRT_OP_IMPL(ctype, onnx_type, func) \
void ort_math_ops_sqrt_##ctype(void* result, const void* input, size_t count) { \
    ctype* res = (ctype*)result; \
    const ctype* data = (const ctype*)input; \
    for (size_t i = 0; i < count; i++) { \
        res[i] = func(data[i]); \
    } \
}

#define ORT_MATH_SQRT_IMPL(c_type, onnx_type, func) ORT_MATH_SQRT_OP_IMPL(c_type, onnx_type, func)
ORT_MATH_FOREACH_NUMERIC_TYPE_EX(ORT_MATH_SQRT_IMPL, sqrt)

/* Square root */
static ort_math_unary_op_func_t ort_math_ops_get_sqrt_func(ONNXTensorElementDataType type) {
    /* Check for SIMD optimization first */
    const ort_math_type_dispatch_t* dispatch = ort_math_get_dispatch(type);
    if (dispatch && dispatch->sqrt_simd_func) {
        return dispatch->sqrt_simd_func;
    }

    /* Fall back to scalar implementation */
    switch (type) {
        #define ORT_MATH_SQRT_CASE(c_type, onnx_type) \
        case onnx_type: return ort_math_ops_sqrt_##c_type;
        ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_SQRT_CASE)
        #undef ORT_MATH_SQRT_CASE
        default: return NULL;
    }
}

/* Floor and round functions */
void ort_math_ops_floor_float(void* result, const void* input, size_t count) {
    float* res = (float*)result;
    const float* val = (const float*)input;
    for (size_t i = 0; i < count; i++) {
        res[i] = floorf(val[i]);
    }
}

void ort_math_ops_floor_double(void* result, const void* input, size_t count) {
    double* res = (double*)result;
    const double* val = (const double*)input;
    for (size_t i = 0; i < count; i++) {
        res[i] = floor(val[i]);
    }
}

void ort_math_ops_round_float(void* result, const void* input, size_t count) {
    float* res = (float*)result;
    const float* data = (const float*)input;
    for (size_t i = 0; i < count; i++) {
        res[i] = roundf(data[i]);
    }
}

void ort_math_ops_round_double(void* result, const void* input, size_t count) {
    double* res = (double*)result;
    const double* data = (const double*)input;
    for (size_t i = 0; i < count; i++) {
        res[i] = round(data[i]);
    }
}

static ort_math_unary_op_func_t ort_math_ops_get_floor_func(ONNXTensorElementDataType type) {
    /* Check for SIMD optimization first */
    const ort_math_type_dispatch_t* dispatch = ort_math_get_dispatch(type);
    if (dispatch && dispatch->floor_simd_func) {
        return dispatch->floor_simd_func;
    }
    
    /* Fall back to scalar implementation */
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return ort_math_ops_floor_float;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return ort_math_ops_floor_double;
        default: return NULL;
    }
}

static ort_math_unary_op_func_t ort_math_ops_get_round_func(ONNXTensorElementDataType type) {
    /* Check for SIMD optimization first */
    const ort_math_type_dispatch_t* dispatch = ort_math_get_dispatch(type);
    if (dispatch && dispatch->round_simd_func) {
        return dispatch->round_simd_func;
    }
    
    /* Fall back to scalar implementation */
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return ort_math_ops_round_float;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return ort_math_ops_round_double;
        default: return NULL;
    }
}

/* Trigonometric functions (floating point only) */
#define ORT_MATH_TRIG_FUNC(func_name, math_func_f, math_func_d) \
void ort_math_ops_##func_name##_float(void* result, const void* input, size_t count) { \
    float* res = (float*)result; \
    const float* val = (const float*)input; \
    for (size_t i = 0; i < count; i++) { \
        res[i] = math_func_f(val[i]); \
    } \
} \
void ort_math_ops_##func_name##_double(void* result, const void* input, size_t count) { \
    double* res = (double*)result; \
    const double* val = (const double*)input; \
    for (size_t i = 0; i < count; i++) { \
        res[i] = math_func_d(val[i]); \
    } \
} \
static ort_math_unary_op_func_t ort_math_ops_get_##func_name##_func(ONNXTensorElementDataType type) { \
    switch (type) { \
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return ort_math_ops_##func_name##_float; \
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return ort_math_ops_##func_name##_double; \
        default: return NULL; \
    } \
}

/* Generate all mathematical functions */
ORT_MATH_TRIG_FUNC(sin, sinf, sin)
ORT_MATH_TRIG_FUNC(cos, cosf, cos)
ORT_MATH_TRIG_FUNC(tan, tanf, tan)
ORT_MATH_TRIG_FUNC(asin, asinf, asin)
ORT_MATH_TRIG_FUNC(acos, acosf, acos)
ORT_MATH_TRIG_FUNC(atan, atanf, atan)
ORT_MATH_TRIG_FUNC(sinh, sinhf, sinh)
ORT_MATH_TRIG_FUNC(cosh, coshf, cosh)
ORT_MATH_TRIG_FUNC(tanh, tanhf, tanh)
ORT_MATH_TRIG_FUNC(exp, expf, exp)
ORT_MATH_TRIG_FUNC(exp2, exp2f, exp2)
ORT_MATH_TRIG_FUNC(log, logf, log)
ORT_MATH_TRIG_FUNC(log2, log2f, log2)
ORT_MATH_TRIG_FUNC(log10, log10f, log10)
ORT_MATH_TRIG_FUNC(ceil, ceilf, ceil)
ORT_MATH_TRIG_FUNC(cbrt, cbrtf, cbrt)
ORT_MATH_TRIG_FUNC(abs, fabsf, fabs)

/* Reciprocal function (special implementation) */
void ort_math_ops_reciprocal_float(void* result, const void* input, size_t count) {
    float* res = (float*)result;
    const float* val = (const float*)input;
    for (size_t i = 0; i < count; i++) {
        res[i] = 1.0f / val[i];
    }
}

void ort_math_ops_reciprocal_double(void* result, const void* input, size_t count) {
    double* res = (double*)result;
    const double* val = (const double*)input;
    for (size_t i = 0; i < count; i++) {
        res[i] = 1.0 / val[i];
    }
}

static ort_math_unary_op_func_t ort_math_ops_get_reciprocal_func(ONNXTensorElementDataType type) {
    /* Check for SIMD optimization first */
    const ort_math_type_dispatch_t* dispatch = ort_math_get_dispatch(type);
    if (dispatch && dispatch->reciprocal_simd_func) {
        return dispatch->reciprocal_simd_func;
    }
    
    /* Fall back to scalar implementation */
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return ort_math_ops_reciprocal_float;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return ort_math_ops_reciprocal_double;
        default: return NULL;
    }
}

/* Sign function (special implementation) */
void ort_math_ops_sign_float(void* result, const void* input, size_t count) {
    float* res = (float*)result;
    const float* val = (const float*)input;
    for (size_t i = 0; i < count; i++) {
        res[i] = (val[i] > 0.0f) ? 1.0f : (val[i] < 0.0f) ? -1.0f : 0.0f;
    }
}

void ort_math_ops_sign_double(void* result, const void* input, size_t count) {
    double* res = (double*)result;
    const double* val = (const double*)input;
    for (size_t i = 0; i < count; i++) {
        res[i] = (val[i] > 0.0) ? 1.0 : (val[i] < 0.0) ? -1.0 : 0.0;
    }
}

static ort_math_unary_op_func_t ort_math_ops_get_sign_func(ONNXTensorElementDataType type) {
    /* Check for SIMD optimization first */
    const ort_math_type_dispatch_t* dispatch = ort_math_get_dispatch(type);
    if (dispatch && dispatch->sign_simd_func) {
        return dispatch->sign_simd_func;
    }
    
    /* Fall back to scalar implementation */
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return ort_math_ops_sign_float;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return ort_math_ops_sign_double;
        default: return NULL;
    }
}

/* =============================================================================
 * PUBLIC INTERFACE FUNCTIONS FOR UNARY MATHEMATICAL OPERATIONS
 * =============================================================================
 */

ort_math_result_t* ort_math_result_sqrt(ort_tensor_t* tensor) {
    if (!ort_math_validate_input(tensor, "sqrt")) {
        return NULL;
    }
    
    ort_math_unary_op_func_t op_func = ort_math_ops_get_sqrt_func(tensor->type);
    if (!op_func) {
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,
            "sqrt: unsupported data type for mathematical function");
        return NULL;
    }
    
    return ort_math_result_element_wise_unary(tensor, op_func, "sqrt");
}

/* Unary function implementations */
#define ORT_MATH_UNARY_RESULT_IMPL(func_name) \
ort_math_result_t* ort_math_result_##func_name(ort_tensor_t* tensor) { \
    if (!ort_math_validate_input(tensor, #func_name)) { \
        return NULL; \
    } \
    ort_math_unary_op_func_t op_func = ort_math_ops_get_##func_name##_func(tensor->type); \
    if (!op_func) { \
        php_ort_status_throw(php_ort_status_math_invalidtype_ce, \
            #func_name ": unsupported data type for mathematical function"); \
        return NULL; \
    } \
    return ort_math_result_element_wise_unary(tensor, op_func, #func_name); \
}

/* Generate all unary function implementations */
ORT_MATH_UNARY_RESULT_IMPL(sin)
ORT_MATH_UNARY_RESULT_IMPL(cos)
ORT_MATH_UNARY_RESULT_IMPL(tan)
ORT_MATH_UNARY_RESULT_IMPL(asin)
ORT_MATH_UNARY_RESULT_IMPL(acos)
ORT_MATH_UNARY_RESULT_IMPL(atan)
ORT_MATH_UNARY_RESULT_IMPL(sinh)
ORT_MATH_UNARY_RESULT_IMPL(cosh)
ORT_MATH_UNARY_RESULT_IMPL(tanh)
ORT_MATH_UNARY_RESULT_IMPL(exp)
ORT_MATH_UNARY_RESULT_IMPL(exp2)
ORT_MATH_UNARY_RESULT_IMPL(log)
ORT_MATH_UNARY_RESULT_IMPL(log2)
ORT_MATH_UNARY_RESULT_IMPL(log10)
ORT_MATH_UNARY_RESULT_IMPL(ceil)
ORT_MATH_UNARY_RESULT_IMPL(floor)
ORT_MATH_UNARY_RESULT_IMPL(round)
ORT_MATH_UNARY_RESULT_IMPL(cbrt)
ORT_MATH_UNARY_RESULT_IMPL(abs)
ORT_MATH_UNARY_RESULT_IMPL(sign)
ORT_MATH_UNARY_RESULT_IMPL(reciprocal)

/* =============================================================================
 * COMPLEX OPERATIONS
 * =============================================================================
 */

ort_math_result_t* ort_math_result_matmul(ort_tensor_t* matrix_a, ort_tensor_t* matrix_b) {
    if (!ort_math_validate_input(matrix_a, "matmul") || !ort_math_validate_input(matrix_b, "matmul")) {
        return NULL;
    }

    // Both tensors must have at least 2 dimensions (last two are the matrix dimensions)
    if (matrix_a->dimensions < 2 || matrix_b->dimensions < 2) {
        php_ort_status_throw(php_ort_status_math_error_ce, 
            "matmul: tensors must have at least 2 dimensions");
        return NULL;
    }
    
    // Get matrix dimensions (last two dimensions)
    size_t a_rows = matrix_a->shape[matrix_a->dimensions - 2];
    size_t a_cols = matrix_a->shape[matrix_a->dimensions - 1];
    size_t b_rows = matrix_b->shape[matrix_b->dimensions - 2];
    size_t b_cols = matrix_b->shape[matrix_b->dimensions - 1];
    
    // Check matrix compatibility: A[..., i, k] x B[..., k, j] -> C[..., i, j]
    if (a_cols != b_rows) {
        php_ort_status_throw(php_ort_status_math_error_ce, 
            "matmul: incompatible matrix dimensions (%zu, %zu) x (%zu, %zu)", 
            a_rows, a_cols, b_rows, b_cols);
        return NULL;
    }

    // Check batch dimensions compatibility for N-D tensors
    size_t batch_dims = matrix_a->dimensions - 2;
    if (matrix_b->dimensions - 2 != batch_dims) {
        php_ort_status_throw(php_ort_status_math_error_ce, 
            "matmul: batch dimensions must match (%zu vs %zu)", 
            batch_dims, matrix_b->dimensions - 2);
        return NULL;
    }
    
    // For N-D tensors, check that all batch dimensions are compatible
    size_t batch_size = 1;
    for (size_t i = 0; i < batch_dims; i++) {
        if (matrix_a->shape[i] != matrix_b->shape[i]) {
            php_ort_status_throw(php_ort_status_math_error_ce, 
                "matmul: batch dimension %zu mismatch (%zd vs %zd)", 
                i, matrix_a->shape[i], matrix_b->shape[i]);
            return NULL;
        }
        batch_size *= matrix_a->shape[i];
    }
    
    // Create result shape: same batch dimensions + (a_rows, b_cols)
    int64_t* result_shape = ecalloc(matrix_a->dimensions, sizeof(int64_t));
    for (size_t i = 0; i < batch_dims; i++) {
        result_shape[i] = matrix_a->shape[i];
    }
    result_shape[matrix_a->dimensions - 2] = a_rows;
    result_shape[matrix_a->dimensions - 1] = b_cols;
    
    // Allocate result tensor
    ort_tensor_t* result = ort_tensor_create_result(result_shape, matrix_a->dimensions, matrix_a->type, "matmul_result");
    efree(result_shape);
    if (!result) {
        return NULL;
    }
    
    // Perform batched matrix multiplication
    float* data_a = (float*)matrix_a->data;
    float* data_b = (float*)matrix_b->data;
    float* data_result = (float*)result->data;
    
    size_t matrix_size_a = a_rows * a_cols;
    size_t matrix_size_b = b_rows * b_cols;
    size_t matrix_size_result = a_rows * b_cols;
    
    // Process each batch
    for (size_t batch = 0; batch < batch_size; batch++) {
        float* batch_a = data_a + batch * matrix_size_a;
        float* batch_b = data_b + batch * matrix_size_b;
        float* batch_result = data_result + batch * matrix_size_result;
        
        // Standard matrix multiplication for this batch
        for (size_t i = 0; i < a_rows; i++) {
            for (size_t j = 0; j < b_cols; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < a_cols; k++) {
                    float a_val = batch_a[i * a_cols + k];
                    float b_val = batch_b[k * b_cols + j];
                    sum += a_val * b_val;
                }
                batch_result[i * b_cols + j] = sum;
            }
        }
    }
    
    return ort_math_result_create(result);
}

ort_math_result_t* ort_math_result_sum(ort_tensor_t* tensor, zval* axis_zval, zend_bool keepdims) {
    if (!ort_math_validate_input(tensor, "sum")) {
        return NULL;
    }

    size_t element_count = tensor->elements;
    if (element_count == 0) {
        php_ort_status_throw(php_ort_status_math_error_ce, "sum: cannot sum empty tensor");
        return NULL;
    }

    // If no axis specified, sum all elements
    if (axis_zval == NULL || Z_TYPE_P(axis_zval) == IS_NULL) {
        ort_tensor_t* result_tensor = ort_tensor_create_result(NULL, 0, tensor->type, "sum_result");
        if (!result_tensor) {
            return NULL;
        }

        switch (tensor->type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                float sum = 0.0f;
                float* data = (float*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(float*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
                double sum = 0.0;
                double* data = (double*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(double*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
                int8_t sum = 0;
                int8_t* data = (int8_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(int8_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
                int16_t sum = 0;
                int16_t* data = (int16_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(int16_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                int32_t sum = 0;
                int32_t* data = (int32_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(int32_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                int64_t sum = 0;
                int64_t* data = (int64_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(int64_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                uint8_t sum = 0;
                uint8_t* data = (uint8_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(uint8_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
                uint16_t sum = 0;
                uint16_t* data = (uint16_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(uint16_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
                uint32_t sum = 0;
                uint32_t* data = (uint32_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(uint32_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
                zend_bool sum = 0;
                zend_bool* data = (zend_bool*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum = sum || data[i]; // Logical OR for boolean sum
                }
                *(zend_bool*)result_tensor->data = sum;
                break;
            }
            default:
                php_ort_status_throw(php_ort_status_math_invalidtype_ce,
                    "sum: unsupported data type for reduction operation");
                return NULL;
        }

        return ort_math_result_create(result_tensor);
    }

    // Handle specified axis
    if (Z_TYPE_P(axis_zval) != IS_LONG) {
        php_ort_status_throw(php_ort_status_tensor_invalidshape_ce, 
            "sum: axis must be an integer");
        return NULL;
    }

    int64_t axis = Z_LVAL_P(axis_zval);
    if (!ort_math_validate_axis(tensor, axis, "sum")) {
        return NULL;
    }

    // Normalize negative axis
    if (axis < 0) {
        axis += tensor->dimensions;
    }

    // Calculate result shape
    int64_t* result_shape = ecalloc(tensor->dimensions - (keepdims ? 0 : 1), sizeof(int64_t));
    size_t result_dims = 0;
    
    for (size_t i = 0; i < tensor->dimensions; i++) {
        if (i != (size_t)axis) {
            result_shape[result_dims++] = tensor->shape[i];
        } else if (keepdims) {
            result_shape[result_dims++] = 1;
        }
    }
    
    if (result_dims == 0 && !keepdims) {
        result_dims = 1;
        result_shape[0] = 1;
    }

    // Create result tensor
    ort_tensor_t* result_tensor = ort_tensor_create_result(result_shape, result_dims, tensor->type, "sum_result");
    if (!result_tensor) {
        efree(result_shape);
        return NULL;
    }

    // Calculate strides
    size_t* strides = ecalloc(tensor->dimensions, sizeof(size_t));
    strides[tensor->dimensions - 1] = 1;
    for (int64_t i = tensor->dimensions - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * tensor->shape[i + 1];
    }

    // Sum along the specified axis
    size_t axis_size = tensor->shape[axis];
    size_t outer_size = 1;
    for (size_t i = 0; i < (size_t)axis; i++) {
        outer_size *= tensor->shape[i];
    }
    size_t inner_size = strides[axis];

    switch (tensor->type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            float* input = (float*)tensor->data;
            float* output = (float*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            double* input = (double*)tensor->data;
            double* output = (double*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    double sum = 0.0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
            int8_t* input = (int8_t*)tensor->data;
            int8_t* output = (int8_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    int8_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
            int16_t* input = (int16_t*)tensor->data;
            int16_t* output = (int16_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    int16_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            int32_t* input = (int32_t*)tensor->data;
            int32_t* output = (int32_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    int32_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            int64_t* input = (int64_t*)tensor->data;
            int64_t* output = (int64_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    int64_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
            uint8_t* input = (uint8_t*)tensor->data;
            uint8_t* output = (uint8_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    uint8_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
            uint16_t* input = (uint16_t*)tensor->data;
            uint16_t* output = (uint16_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    uint16_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
            uint32_t* input = (uint32_t*)tensor->data;
            uint32_t* output = (uint32_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    uint32_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            zend_bool* input = (zend_bool*)tensor->data;
            zend_bool* output = (zend_bool*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    zend_bool sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum = sum || input[idx]; // Logical OR for boolean sum
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        default:
            php_ort_status_throw(php_ort_status_math_invalidtype_ce,
                "sum: unsupported data type for reduction operation");
            efree(strides);
            efree(result_shape);
            return NULL;
    }

    efree(strides);
    efree(result_shape);

    return ort_math_result_create(result_tensor);
}
