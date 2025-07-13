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

static ort_math_element_op_func_t 
    ort_math_frontend_get_div_func(
        ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->div_func;
}

#define ORT_MATH_FRONTEND_DIV_SCALAR_IMPL(c_type, onnx_type) \
    ORT_MATH_FRONTEND_SCALAR_OP_IMPL(div, c_type, /)
ORT_MATH_FOREACH_REAL_TYPE(
    ORT_MATH_FRONTEND_DIV_SCALAR_IMPL)
#undef ORT_MATH_FRONTEND_DIV_SCALAR_IMPL

static ort_math_scalar_op_func_t 
    ort_math_frontend_get_div_scalar_func(
        ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->div_scalar_func;
}

ORT_MATH_RESULT_BINARY_IMPL(divide,   ort_math_frontend_get_div_func, &ort_math_promotion_schema_div)
ORT_MATH_RESULT_SCALAR_IMPL(divide,   ort_math_frontend_get_div_scalar_func, &ort_math_promotion_schema_div)