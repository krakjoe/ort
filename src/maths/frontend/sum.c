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
#include "maths/result.h"

#define ORT_MATH_SUM_AXIS_IMPL_FOR_TYPE(c_type)             \
    ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(sum, c_type) { \
        c_type* va = (c_type*)a;                            \
        c_type* res = (c_type*)result;                      \
        for (size_t i = 0; i < outer; i++) {                \
            for (size_t k = 0; k < inner; k++) {            \
                c_type sum = 0;                             \
                for (size_t j = 0; j < axis; j++) {         \
                    size_t idx =                            \
                        i * (axis * inner) + j * inner + k; \
                    sum += va[idx];                         \
                }                                           \
                res[i * inner + k] = sum;                   \
            }                                               \
        }                                                   \
    }

#define ORT_MATH_SUM_IMPL_FOR_TYPE(c_type)             \
    ORT_MATH_FRONTEND_REDUCTION_OP_DECL(sum, c_type) { \
        c_type* va = (c_type*)a;                       \
        c_type* res = (c_type*)result;                 \
        c_type sum = 0;                                \
        for (size_t idx = 0; idx < count; idx++) {     \
            sum += va[idx];                            \
        }                                              \
        res[0] = sum;                                  \
    }

ORT_MATH_SUM_AXIS_IMPL_FOR_TYPE(float)
ORT_MATH_SUM_AXIS_IMPL_FOR_TYPE(double)
ORT_MATH_SUM_AXIS_IMPL_FOR_TYPE(int64_t)

ORT_MATH_SUM_IMPL_FOR_TYPE(float)
ORT_MATH_SUM_IMPL_FOR_TYPE(double)
ORT_MATH_SUM_IMPL_FOR_TYPE(int64_t)

static zend_always_inline ONNXTensorElementDataType ort_math_frontend_sum_get_promotion_schema(ONNXTensorElementDataType type) {
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL   ||
        type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8   ||
        type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8  ||
        type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16  ||
        type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ||
        type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32  ||
        type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    return type; // FLOAT, DOUBLE, INT64, etc.
}

static ort_math_unary_op_func_t ort_math_frontend_get_reduce_tensor_sum(ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->sum_func;
}

ORT_MATH_REDUCE_TENSOR_PROMOTE_RESULT_IMPL(sum,
    ort_math_frontend_get_reduce_tensor_sum,
    ort_math_validate_input,
    ort_math_frontend_sum_get_promotion_schema)

static ort_math_reduction_op_func_t ort_math_frontend_get_reduce_axis_sum(ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->sum_axis_func;
}

ORT_MATH_REDUCE_AXIS_PROMOTE_RESULT_IMPL(sum,
    ort_math_frontend_get_reduce_axis_sum,
    ort_math_validate_input,
    ort_math_validate_axis,
    ort_math_result_reduce,
    ort_math_frontend_sum_get_promotion_schema)