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
 @brief Implements frontend div operations for tensors
 @test tests/math/div
*/

#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/schema/div.h"

#define ORT_MATH_FRONTEND_DIV_IMPL(c_type, onnx_type) \
    ORT_MATH_FRONTEND_BINARY_OP_IMPL(div, c_type, /)
ORT_MATH_FOREACH_REAL_TYPE(
    ORT_MATH_FRONTEND_DIV_IMPL)
#undef ORT_MATH_FRONTEND_DIV_IMPL

#define ORT_MATH_FRONTEND_DIV_SCALAR_IMPL(c_type, onnx_type) \
    ORT_MATH_FRONTEND_SCALAR_OP_IMPL(div, c_type, /)
ORT_MATH_FOREACH_REAL_TYPE(
    ORT_MATH_FRONTEND_DIV_SCALAR_IMPL)
#undef ORT_MATH_FRONTEND_DIV_SCALAR_IMPL

ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(
    ort_math_kernel_binary_t, div)
ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(
    ort_math_kernel_scalar_t, div_scalar)

ORT_MATH_RESULT_BINARY_IMPL(divide,
    ORT_MATH_FRONTEND_DISPATCH_SYMBOL(div),
    &ort_math_promotion_schema_div)

ORT_MATH_RESULT_SCALAR_IMPL(divide,
    ORT_MATH_FRONTEND_DISPATCH_SYMBOL(div_scalar),
    &ort_math_promotion_schema_div)