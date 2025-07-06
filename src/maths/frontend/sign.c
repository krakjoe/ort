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

void ort_math_ops_sign_float(void* result, const void* a, size_t count) {
    float* res = (float*)result;
    const float* va = (const float*)a;
    for (size_t i = 0; i < count; i++) {
        res[i] = (va[i] > 0.0f) ? 
            1.0f : (va[i] < 0.0f) ? -1.0f : 0.0f;
    }
}

void ort_math_ops_sign_double(void* result, const void* a, size_t count) {
    double* res = (double*)result;
    const double* va = (const double*)a;
    for (size_t i = 0; i < count; i++) {
        res[i] = (va[i] > 0.0) ? 
            1.0 : (va[i] < 0.0) ? -1.0 : 0.0;
    }
}

static ort_math_unary_op_func_t ort_math_ops_get_sign_func(ONNXTensorElementDataType type) {    
    const ort_math_type_dispatch_t* dispatch =
        ort_math_get_dispatch(type);
    return dispatch->sign_func;
}

ORT_MATH_UNARY_RESULT_IMPL(sign, ort_math_ops_get_sign_func)