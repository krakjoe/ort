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

/*
 @brief Implements frontend reciprocal operations for tensors
 @test tests/math/recip
*/

#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/schema/recip.h"

/* =============================================================================
 * RECIP OPERATIONS
 * =============================================================================
 */

ORT_MATH_FRONTEND_UNARY_OP_DECL(recip, float) {
    float* res = (float*)result;
    const float* va = (const float*)a;
    for (size_t i = 0; i < count; i++) {
        res[i] = 1.0f / va[i];
    }
}

ORT_MATH_FRONTEND_UNARY_OP_DECL(recip, double) {
    double* res = (double*)result;
    const double* va = (const double*)a;
    for (size_t i = 0; i < count; i++) {
        res[i] = 1.0 / va[i];
    }
}

static ort_math_unary_op_func_t
    ort_math_frontend_get_recip_func(ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->recip_func;
}

ORT_MATH_UNARY_RESULT_IMPL(recip, ort_math_frontend_get_recip_func, &ort_math_promotion_schema_recip)