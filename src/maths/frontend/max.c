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

#define ORT_MATH_MAX_AXIS_IMPL_FOR_TYPE(c_type, unused) \
    void ort_math_frontend_max_axis_##c_type( \
        void* result, const void *a, size_t outer, size_t axis, size_t inner) { \
        c_type* va = (c_type*)a; \
        c_type* res = (c_type*)result; \
        for (size_t i = 0; i < outer; i++) { \
            for (size_t k = 0; k < inner; k++) { \
                c_type max = va[i * (axis * inner) + 0 * inner + k]; \
                for (size_t j = 1; j < axis; j++) { \
                    size_t idx =  \
                        i * (axis * inner) + j * inner + k; \
                    if (va[idx] > max) max = va[idx]; \
                } \
                res[i * inner + k] = max; \
            } \
        } \
    }

void ort_math_frontend_max_axis_zend_bool(
    void* result, const void *a, size_t outer, size_t axis, size_t inner) {
    zend_bool* va = (zend_bool*)a;
    zend_bool* res = (zend_bool*)result;
    for (size_t i = 0; i < outer; i++) {
        for (size_t k = 0; k < inner; k++) {
            zend_bool max = 0;
            for (size_t j = 0; j < axis; j++) {
                size_t idx =
                    i * (axis * inner) + j * inner + k;
                max = max || va[idx];
            }
            res[i * inner + k] = max;
        }
    }
}

#define ORT_MATH_MAX_IMPL_FOR_TYPE(c_type, unused) \
    void ort_math_frontend_max_##c_type( \
        void* result, const void *a, size_t count) { \
        c_type* va = (c_type*)a; \
        c_type* res = (c_type*)result; \
        c_type max = va[0]; \
        for (size_t idx = 1; idx < count; idx++) { \
            if (va[idx] > max) max = va[idx]; \
        } \
        res[0] = max; \
    }

void ort_math_frontend_max_zend_bool(
    void* result, const void *a, size_t count) {
    zend_bool* va = (zend_bool*)a;
    zend_bool* res = (zend_bool*)result;
    zend_bool max = 0;
    for (size_t idx = 0; idx < count; idx++) {
        max = max || va[idx];
    }
    res[0] = max;
}

ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_MAX_AXIS_IMPL_FOR_TYPE)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_MAX_IMPL_FOR_TYPE)

ort_tensor_t* ort_math_result_max(ort_tensor_t* tensor, zval* axis_zval, zend_bool keepdims) {
    if (!ort_math_validate_input(tensor, "max")) {
        return NULL;
    }

    // If no axis specified, max all elements
    if (axis_zval == NULL || Z_TYPE_P(axis_zval) == IS_NULL) {
        ort_tensor_t* result = ort_math_result_tensor(NULL, 0, tensor->type, "max_result");
        const ort_math_dispatch_t* dispatch =
            ort_math_dispatch_type(tensor->type);

        dispatch->max_func(
            result->data, tensor->data, tensor->elements);

        return result;
    }

    // Handle specified axis
    if (Z_TYPE_P(axis_zval) != IS_LONG) {
        php_ort_status_throw(php_ort_status_tensor_invalidshape_ce, 
            "max: axis must be an integer");
        return NULL;
    }

    int64_t axis = Z_LVAL_P(axis_zval);
    if (!ort_math_validate_axis(tensor, axis, "max")) {
        return NULL;
    }

    // Normalize negative axis
    if (axis < 0) {
        axis += tensor->dimensions;
    }

    // Calculate result shape
    int64_t* result_shape = ecalloc(tensor->dimensions - (keepdims ? 0 : 1), sizeof(int64_t));
    size_t result_dims = 0;
    for (size_t i = 0; i < tensor->dimensions; i++) {
        if (i != (size_t)axis) {
            result_shape[result_dims++] = tensor->shape[i];
        } else if (keepdims) {
            result_shape[result_dims++] = 1;
        }
    }

    ort_tensor_t* result = ort_math_result_tensor(result_shape, result_dims, tensor->type, "max_result");

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

    dispatch->max_axis_func(
        result->data, tensor->data,
        outer_size, axis_size, inner_size);

    efree(strides);
    efree(result_shape);

    return result;
}
