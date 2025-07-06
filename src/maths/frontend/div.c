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

#define ORT_MATH_DIV_IMPL(c_type, onnx_type) \
    ORT_MATH_BINARY_OP_IMPL(div, c_type, onnx_type, /)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_DIV_IMPL)
ORT_MATH_BINARY_OP_IMPL(div, \
    zend_bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, &&)

static ort_math_element_op_func_t 
    ort_math_ops_get_div_func(ONNXTensorElementDataType type) {
    switch (type) {
#define ORT_MATH_DIV_CASE(c_type, onnx_type) \
    ORT_MATH_BINARY_FUNC_GETTER_CASE(c_type, onnx_type, div)
        ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_DIV_CASE)
#undef ORT_MATH_DIV_CASE
        default: return NULL;
    }
}

#define ORT_MATH_DIV_SCALAR_IMPL(c_type, onnx_type) \
    ORT_MATH_SCALAR_OP_IMPL(div, c_type, onnx_type, /)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_DIV_SCALAR_IMPL)
ORT_MATH_SCALAR_OP_IMPL(div, \
    zend_bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, &&)

static ort_math_scalar_op_func_t 
    ort_math_ops_get_div_scalar_func(ONNXTensorElementDataType type) {
    switch (type) {
#define ORT_MATH_DIV_SCALAR_CASE(c_type, onnx_type) \
    ORT_MATH_SCALAR_FUNC_GETTER_CASE(c_type, onnx_type, div)
        ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_DIV_SCALAR_CASE)
#undef ORT_MATH_DIV_SCALAR_CASE
        default: return NULL;
    }
}

ORT_MATH_BINARY_RESULT_IMPL(divide,   ort_math_ops_get_div_func)
ORT_MATH_SCALAR_RESULT_IMPL(divide,   ort_math_ops_get_div_scalar_func)