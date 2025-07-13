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
#include "maths/schema/mul.h"

#define ORT_MATH_MUL_IMPL(c_type, onnx_type) \
    ORT_MATH_BINARY_OP_IMPL(mul, c_type, onnx_type, *)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_MUL_IMPL)
ORT_MATH_BINARY_OP_IMPL(mul, \
    zend_bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, &&)

static ort_math_element_op_func_t 
    ort_math_frontend_get_mul_func(ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->mul_func;
}

#define ORT_MATH_MUL_SCALAR_IMPL(c_type, onnx_type) \
    ORT_MATH_SCALAR_OP_IMPL(mul, c_type, onnx_type, *)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_MUL_SCALAR_IMPL)
ORT_MATH_SCALAR_OP_IMPL(mul, \
    zend_bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, &&)

static ort_math_scalar_op_func_t 
    ort_math_frontend_get_mul_scalar_func(ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->mul_scalar_func;
}

ORT_MATH_BINARY_RESULT_WITH_SCHEMA_IMPL(multiply, ort_math_frontend_get_mul_func, &ort_math_promotion_schema_mul)
ORT_MATH_SCALAR_RESULT_WITH_SCHEMA_IMPL(multiply, ort_math_frontend_get_mul_scalar_func, &ort_math_promotion_schema_mul)