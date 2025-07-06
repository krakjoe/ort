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
#include "maths/core.h"

void ort_math_ops_recip_float(void* result, const void* a, size_t count) {
    float* res = (float*)result;
    const float* va = (const float*)a;
    for (size_t i = 0; i < count; i++) {
        res[i] = 1.0f / va[i];
    }
}

void ort_math_ops_recip_double(void* result, const void* a, size_t count) {
    double* res = (double*)result;
    const double* va = (const double*)a;
    for (size_t i = 0; i < count; i++) {
        res[i] = 1.0 / va[i];
    }
}

static ort_math_unary_op_func_t ort_math_ops_get_recip_func(ONNXTensorElementDataType type) {
    /* Check for SIMD optimization first */
    const ort_math_type_dispatch_t* dispatch = ort_math_get_dispatch(type);
    if (dispatch && dispatch->recip_simd_func) {
        return dispatch->recip_simd_func;
    }
    
    /* Fall back to scalar implementation */
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return ort_math_ops_recip_float;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return ort_math_ops_recip_double;
        default: return NULL;
    }
}

ORT_MATH_UNARY_RESULT_IMPL(recip, ort_math_ops_get_recip_func)