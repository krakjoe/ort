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

#include "maths/codegen.h"
#include "maths/dispatch.h"

/* =============================================================================
 * POWER OPERATIONS
 * =============================================================================
 */

#define ORT_MATH_POW_IMPL_FOR_TYPE(c_type, unused)               \
    static zend_always_inline c_type ort_math_pow_impl_##c_type( \
        c_type a, c_type b) {                                    \
    c_type result =    1;                                        \
    if (b < 0) {                                                 \
        return 0;                                                \
    }                                                            \
    while (b) {                                                  \
        if (b & 1) result *= a;                                  \
        a *= a;                                                  \
        b >>= 1;                                                 \
    }                                                            \
    return result;                                               \
}

ORT_MATH_FOREACH_INTEGER_TYPE(ORT_MATH_POW_IMPL_FOR_TYPE)

#define ORT_MATH_POW_IMPL(c_type, onnx_type)                        \
    void ort_math_ops_pow_##c_type(                                 \
        void* result, const void* a, const void* b, size_t count) { \
    c_type* res = (c_type*)result;                                  \
    const c_type* va = (const c_type*)a;                            \
    const c_type* vb = (const c_type*)b;                            \
    for (size_t i = 0; i < count; i++) {                            \
        res[i] = ort_math_pow_impl_##c_type(                        \
            va[i], vb[i]);                                          \
    }                                                               \
}

ORT_MATH_FOREACH_INTEGER_TYPE(ORT_MATH_POW_IMPL)

void ort_math_ops_pow_float(void* result, const void* a, const void* b, size_t count) {
    float* res = (float*)result;
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = powf(va[i], vb[i]);
    }
}

void ort_math_ops_pow_double(void* result, const void* a, const void* b, size_t count) {
    double* res = (double*)result;
    const double* va = (const double*)a;
    const double* vb = (const double*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = pow(va[i], vb[i]);
    }
}

static ort_math_element_op_func_t ort_math_ops_get_pow_func(ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->pow_func;
}

#define ORT_MATH_POW_SCALAR_IMPL(c_type, onnx_type)                   \
void ort_math_ops_pow_scalar_##c_type(                                \
    void* result, const void* a, const void* b, size_t count) {       \
    c_type* res = (c_type*)result;                                    \
    const c_type* va = (const c_type*)a;                              \
    c_type sb = *(const c_type*)b;                                    \
    for (size_t i = 0; i < count; i++) {                              \
        res[i] = ort_math_pow_impl_##c_type(va[i], sb);               \
    }                                                                 \
}

ORT_MATH_FOREACH_INTEGER_TYPE(ORT_MATH_POW_SCALAR_IMPL)

void ort_math_ops_pow_scalar_float(void* result, const void* a, const void* b, size_t count) {
    float* res = (float*)result;
    const float* va = (const float*)a;
    float vb = *(const float*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = powf(va[i], vb);
    }
}

void ort_math_ops_pow_scalar_double(void* result, const void* a, const void* b, size_t count) {
    double* res = (double*)result;
    const double* va = (const double*)a;
    double sb = *(const double*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = pow(va[i], sb);
    }
}

static ort_math_scalar_op_func_t ort_math_ops_get_pow_scalar_func(ONNXTensorElementDataType type) {
    switch (type) {
#define ORT_MATH_POW_SCALAR_CASE(c_type, onnx_type) \
    ORT_MATH_SCALAR_FUNC_GETTER_CASE(c_type, onnx_type, pow)
        ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_POW_SCALAR_CASE)
#undef ORT_MATH_POW_SCALAR_CASE
        default: return NULL;
    }
}

ORT_MATH_BINARY_RESULT_IMPL(pow, ort_math_ops_get_pow_func)
ORT_MATH_SCALAR_RESULT_IMPL(pow, ort_math_ops_get_pow_scalar_func)