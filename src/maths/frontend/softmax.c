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

#define ORT_MATH_SOFTMAX_AXIS_IMPL_FOR_TYPE(c_type, unused) \
    ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(softmax, c_type) { \
        c_type* va = (c_type*)a; \
        c_type* res = (c_type*)result; \
        /*
         * This implementation computes softmax along the axis, preserving the input shape.
         * For each slice along the axis, compute softmax and write back to the same positions.
         * outer: product of dims before axis
         * axis: size of axis
         * inner: product of dims after axis
         */ \
        for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) { \
            for (size_t inner_idx = 0; inner_idx < inner; ++inner_idx) { \
                /* Find max for numerical stability */ \
                c_type max = va[outer_idx * axis * inner + 0 * inner + inner_idx]; \
                for (size_t axis_idx = 1; axis_idx < axis; ++axis_idx) { \
                    size_t idx = outer_idx * axis * inner + axis_idx * inner + inner_idx; \
                    if (va[idx] > max) max = va[idx]; \
                } \
                /* Compute exp and sum */ \
                c_type sum = 0; \
                for (size_t axis_idx = 0; axis_idx < axis; ++axis_idx) { \
                    size_t idx = outer_idx * axis * inner + axis_idx * inner + inner_idx; \
                    res[idx] = (c_type)exp((double)va[idx] - (double)max); \
                    sum += res[idx]; \
                } \
                /* Normalize */ \
                for (size_t axis_idx = 0; axis_idx < axis; ++axis_idx) { \
                    size_t idx = outer_idx * axis * inner + axis_idx * inner + inner_idx; \
                    res[idx] /= sum; \
                } \
            } \
        } \
    }

ORT_MATH_FOREACH_REAL_TYPE(ORT_MATH_SOFTMAX_AXIS_IMPL_FOR_TYPE)

static zend_always_inline ONNXTensorElementDataType ort_math_frontend_softmax_get_promotion_schema(ONNXTensorElementDataType type) {
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        return type;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; // or trigger error upstream
}

static ort_math_reduction_op_func_t ort_math_frontend_get_reduce_axis_softmax(ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    if (!dispatch) {
        return NULL;
    }
    
    return dispatch->softmax_axis_func;
}

ORT_MATH_REDUCE_AXIS_PROMOTE_RESULT_IMPL(softmax,
    ort_math_frontend_get_reduce_axis_softmax,
    ort_math_validate_input,
    ort_math_validate_axis,
    ort_math_result_reduce,
    ort_math_frontend_softmax_get_promotion_schema)