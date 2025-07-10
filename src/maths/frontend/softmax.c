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

ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(softmax, zend_bool) {
    zend_bool* va = (zend_bool*)a;
    zend_bool* res = (zend_bool*)result;
    for (size_t i = 0; i < outer; i++) {
        for (size_t k = 0; k < inner; k++) {
            zend_bool any = 0;
            for (size_t j = 0; j < axis; j++) {
                size_t idx =
                    i * (axis * inner) + j * inner + k;
                any = any || va[idx];
            }
            for (size_t j = 0; j < axis; j++) {
                size_t idx = i * (axis * inner) + j * inner + k;
                res[idx] = any ? 1 : 0;
            }
        }
    }
}

ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_SOFTMAX_AXIS_IMPL_FOR_TYPE)

ort_tensor_t* ort_math_result_softmax(ort_tensor_t* tensor, zval* axis_zval, zend_bool keepdims) {
    if (!ort_math_validate_input(tensor, "softmax")) {
        return NULL;
    }

    int64_t axis;
    if (axis_zval == NULL || Z_TYPE_P(axis_zval) == IS_NULL) {
        // Default to last axis (ONNX/NumPy semantics)
        axis = tensor->dimensions - 1;
    } else {
        if (Z_TYPE_P(axis_zval) != IS_LONG) {
            php_ort_status_throw(php_ort_status_tensor_invalidshape_ce, 
                "softmax: axis must be an integer");
            return NULL;
        }
        axis = Z_LVAL_P(axis_zval);
        if (!ort_math_validate_axis(tensor, axis, "softmax")) {
            return NULL;
        }
        // Normalize negative axis
        if (axis < 0) {
            axis += tensor->dimensions;
        }
    }

    // Calculate result shape: if keepdims, axis dim is 1; else, shape is same as input
    int64_t* result_shape = NULL;
    size_t result_dims = 0;
    if (keepdims) {
        result_shape = ecalloc(tensor->dimensions, sizeof(int64_t));
        for (size_t i = 0; i < tensor->dimensions; i++) {
            result_shape[i] = tensor->shape[i];
        }
        result_shape[axis] = 1;
        result_dims = tensor->dimensions;
    } else {
        result_shape = ecalloc(tensor->dimensions, sizeof(int64_t));
        for (size_t i = 0; i < tensor->dimensions; i++) {
            result_shape[i] = tensor->shape[i];
        }
        result_dims = tensor->dimensions;
    }
    ort_tensor_t* result = ort_math_result_tensor(result_shape, result_dims, tensor->type, "softmax_result");

    // Calculate strides
    size_t* strides = ecalloc(tensor->dimensions, sizeof(size_t));
    strides[tensor->dimensions - 1] = 1;
    for (int64_t i = tensor->dimensions - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * tensor->shape[i + 1];
    }

    size_t axis_size = tensor->shape[axis];
    size_t outer_size = 1;
    for (size_t i = 0; i < (size_t)axis; i++) {
        outer_size *= tensor->shape[i];
    }
    size_t inner_size = strides[axis];

    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(tensor->type);

    dispatch->softmax_axis_func(
        result->data, tensor->data,
        outer_size, axis_size, inner_size);

    efree(strides);
    efree(result_shape);

    return result;
}
