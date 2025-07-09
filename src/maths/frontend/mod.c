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
 * MODULO OPERATIONS
 * =============================================================================
 */
void ort_math_frontend_mod_float(void* result, const void* a, const void* b, size_t count) {
    float* res = (float*)result;
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = fmodf(va[i], vb[i]);
    }
}

void ort_math_frontend_mod_double(void* result, const void* a, const void* b, size_t count) {
    double* res = (double*)result;
    const double* va = (const double*)a;
    const double* vb = (const double*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = fmod(va[i], vb[i]);
    }
}

#define ORT_MATH_MOD_IMPL(c_type, onnx_type) \
    ORT_MATH_BINARY_OP_IMPL(mod, c_type, onnx_type, %)
ORT_MATH_FOREACH_INTEGER_TYPE(ORT_MATH_MOD_IMPL)

static ort_math_element_op_func_t ort_math_frontend_get_mod_func(ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->mod_func;
}

void ort_math_frontend_mod_scalar_float(void* result, const void* a, const void* b, size_t count) {
    float* res = (float*)result;
    const float* va = (const float*)a;
    float sb = *(const float*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = fmodf(va[i], sb);
    }
}

void ort_math_frontend_mod_scalar_double(void* result, const void* a, const void* b, size_t count) {
    double* res = (double*)result;
    const double* va = (const double*)a;
    double sb = *(const double*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = fmod(va[i], sb);
    }
}

#define ORT_MATH_MOD_SCALAR_IMPL(c_type, onnx_type) \
    ORT_MATH_SCALAR_OP_IMPL(mod, c_type, onnx_type, %)
ORT_MATH_FOREACH_INTEGER_TYPE(ORT_MATH_MOD_SCALAR_IMPL)

static ort_math_scalar_op_func_t ort_math_frontend_get_mod_scalar_func(ONNXTensorElementDataType type) {
    switch (type) {
#define ORT_MATH_MOD_SCALAR_CASE(c_type, onnx_type) \
    ORT_MATH_SCALAR_FUNC_GETTER_CASE(c_type, onnx_type, mod)
        ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_MOD_SCALAR_CASE)
#undef ORT_MATH_MOD_SCALAR_CASE
        default: return NULL;
    }
}

ORT_MATH_BINARY_RESULT_IMPL(mod, ort_math_frontend_get_mod_func)
ORT_MATH_SCALAR_RESULT_IMPL(mod, ort_math_frontend_get_mod_scalar_func)