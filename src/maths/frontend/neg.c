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
 @brief Implements frontend negation operations for tensors
 @test tests/math/neg
*/

#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/schema/neg.h"

#define ORT_MATH_NEG_IMPL(c_type, onnx_type)   \
ORT_MATH_FRONTEND_UNARY_OP_DECL(neg, c_type) { \
    c_type* res = (c_type*)result;             \
    const c_type* va = (const c_type*)a;       \
    for (size_t i = 0; i < count; i++) {       \
        res[i] = -va[i];                       \
    }                                          \
}

ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_NEG_IMPL)

ORT_MATH_FRONTEND_UNARY_OP_DECL(neg, zend_bool) {
    zend_bool* res = (zend_bool*)result;
    const zend_bool* va = (const zend_bool*)a;
    for (size_t i = 0; i < count; i++) {
        res[i] = !va[i];
    }
}

static ort_math_unary_op_func_t ort_math_frontend_get_neg_func(ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->neg_func;
}

ORT_MATH_UNARY_RESULT_IMPL(neg, ort_math_frontend_get_neg_func, &ort_math_promotion_schema_neg)