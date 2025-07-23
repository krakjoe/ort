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
 @brief Implements frontend add operations for tensors
 @test tests/math/add
*/

#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/schema/add.h"

#define ORT_MATH_FRONTEND_ADD_IMPL(c_type, onnx_type) \
    ORT_MATH_FRONTEND_BINARY_OP_IMPL(add, c_type, +)
ORT_MATH_FOREACH_NUMERIC_TYPE(
    ORT_MATH_FRONTEND_ADD_IMPL)
#undef ORT_MATH_FRONTEND_ADD_IMPL
ORT_MATH_FRONTEND_BINARY_OP_IMPL(add, zend_bool, ||)

#define ORT_MATH_FRONTEND_ADD_SCALAR_IMPL(c_type, onnx_type) \
    ORT_MATH_FRONTEND_SCALAR_OP_IMPL(add, c_type, +)
ORT_MATH_FOREACH_NUMERIC_TYPE(
    ORT_MATH_FRONTEND_ADD_SCALAR_IMPL)
#undef ORT_MATH_FRONTEND_ADD_SCALAR_IMPL
ORT_MATH_FRONTEND_SCALAR_OP_IMPL(add, zend_bool, ||)

ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(
    ort_math_element_op_func_t, add)
ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(
    ort_math_scalar_op_func_t, add_scalar)

ORT_MATH_RESULT_BINARY_IMPL(add,
    ORT_MATH_FRONTEND_DISPATCH_SYMBOL(add),
    &ort_math_promotion_schema_add)
ORT_MATH_RESULT_SCALAR_IMPL(add,
    ORT_MATH_FRONTEND_DISPATCH_SYMBOL(add_scalar),
    &ort_math_promotion_schema_add)