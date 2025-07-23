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

#define ORT_MATH_FRONTEND_NEG_IMPL(c_type, onnx_type) \
    ORT_MATH_FRONTEND_UNARY_OP_IMPL(neg, c_type, -)
ORT_MATH_FOREACH_NUMERIC_TYPE(
    ORT_MATH_FRONTEND_NEG_IMPL)
#undef ORT_MATH_FRONTEND_NEG_IMPL

ORT_MATH_FRONTEND_UNARY_OP_IMPL(neg, zend_bool, !)

ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(
    ort_math_unary_op_func_t, neg)

ORT_MATH_RESULT_UNARY_IMPL(neg,
    ORT_MATH_FRONTEND_DISPATCH_SYMBOL(neg),
    &ort_math_promotion_schema_neg)