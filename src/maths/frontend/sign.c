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
#include "maths/schema/sign.h"

ORT_MATH_FRONTEND_UNARY_OP_DECL(sign, float) {
    float* res = (float*)result;
    const float* va = (const float*)a;
    for (size_t i = 0; i < count; i++) {
        res[i] = (va[i] > 0.0f) ? 
            1.0f : (va[i] < 0.0f) ? -1.0f : 0.0f;
    }
}

ORT_MATH_FRONTEND_UNARY_OP_DECL(sign, double) {
    double* res = (double*)result;
    const double* va = (const double*)a;
    for (size_t i = 0; i < count; i++) {
        res[i] = (va[i] > 0.0) ? 
            1.0 : (va[i] < 0.0) ? -1.0 : 0.0;
    }
}

#define ORT_MATH_SIGN_IMPL_FOR_SIGNED(c_type, unused)  \
    ORT_MATH_FRONTEND_UNARY_OP_DECL(sign, c_type) {    \
        c_type* va = (c_type*)a;                       \
        c_type* res = (c_type*)result;                 \
        c_type sum = 0;                                \
        for (size_t idx = 0; idx < count; idx++) {     \
            res[idx] = (va[idx] > 0) ?                 \
                1 :  (va[idx] < 0) ? -1 : 0;           \
        }                                              \
    }
ORT_MATH_FOREACH_SIGNED_TYPE(
    ORT_MATH_SIGN_IMPL_FOR_SIGNED)
#undef ORT_MATH_SIGN_IMPL_FOR_SIGNED

#define ORT_MATH_SIGN_IMPL_FOR_UNSIGNED(c_type, unused) \
    ORT_MATH_FRONTEND_UNARY_OP_DECL(sign, c_type) {     \
        c_type* va = (c_type*)a;                        \
        c_type* res = (c_type*)result;                  \
        c_type sum = 0;                                 \
        for (size_t idx = 0; idx < count; idx++) {      \
            res[idx] = (va[idx] > 0) ? 1 : 0;           \
        }                                               \
    }

ORT_MATH_FOREACH_UNSIGNED_TYPE(
    ORT_MATH_SIGN_IMPL_FOR_UNSIGNED)
#undef ORT_MATH_SIGN_IMPL_FOR_UNSIGNED

static ort_math_unary_op_func_t ort_math_frontend_get_sign_func(ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->sign_func;
}

ORT_MATH_RESULT_UNARY_IMPL(sign, ort_math_frontend_get_sign_func, &ort_math_promotion_schema_sign)