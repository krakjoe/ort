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

#define ORT_MATH_MIN_AXIS_IMPL_FOR_TYPE(c_type, unused) \
    static zend_always_inline void ort_math_min_axis_impl_##c_type( \
        void* result, const void *a, size_t outer, size_t axis, size_t inner) { \
        c_type* va = (c_type*)a; \
        c_type* res = (c_type*)result; \
        for (size_t i = 0; i < outer; i++) { \
            for (size_t k = 0; k < inner; k++) { \
                c_type min = va[i * (axis * inner) + 0 * inner + k]; \
                for (size_t j = 1; j < axis; j++) { \
                    size_t idx =  \
                        i * (axis * inner) + j * inner + k; \
                    if (va[idx] < min) min = va[idx]; \
                } \
                res[i * inner + k] = min; \
            } \
        } \
    }

static zend_always_inline void ort_math_min_axis_impl_zend_bool(
    void* result, const void *a, size_t outer, size_t axis, size_t inner) {
    zend_bool* va = (zend_bool*)a;
    zend_bool* res = (zend_bool*)result;
    for (size_t i = 0; i < outer; i++) {
        for (size_t k = 0; k < inner; k++) {
            zend_bool min = 1;
            for (size_t j = 0; j < axis; j++) {
                size_t idx =
                    i * (axis * inner) + j * inner + k;
                min = min && va[idx];
            }
            res[i * inner + k] = min;
        }
    }
}


#define ORT_MATH_MIN_IMPL_FOR_TYPE(c_type, unused) \
    static zend_always_inline void ort_math_min_impl_##c_type( \
        void* result, const void *a, size_t count) { \
        c_type* va = (c_type*)a; \
        c_type* res = (c_type*)result; \
        c_type min = va[0]; \
        for (size_t idx = 1; idx < count; idx++) { \
            if (va[idx] < min) min = va[idx]; \
        } \
        res[0] = min; \
    }

static zend_always_inline void ort_math_min_impl_zend_bool(
    void* result, const void *a, size_t count) {
    zend_bool* va = (zend_bool*)a;
    zend_bool* res = (zend_bool*)result;
    zend_bool min = 1;
    for (size_t idx = 0; idx < count; idx++) {
        min = min && va[idx];
    }
    res[0] = min;
}

ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_MIN_AXIS_IMPL_FOR_TYPE)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_MIN_IMPL_FOR_TYPE)

ort_tensor_t* ort_math_result_min(ort_tensor_t* tensor, zval* axis_zval, zend_bool keepdims) {
    if (!ort_math_validate_input(tensor, "min")) {
        return NULL;
    }

    size_t element_count = tensor->elements;
    if (element_count == 0) {
        php_ort_status_throw(php_ort_status_math_error_ce, 
            "min: cannot min empty tensor");
        return NULL;
    }

    // If no axis specified, min all elements
    if (axis_zval == NULL || Z_TYPE_P(axis_zval) == IS_NULL) {
        ort_tensor_t* result = ort_math_result_tensor(NULL, 0, tensor->type, "min_result");

        switch (tensor->type) {
            #define ORT_MATH_MIN_CASE(c_type, onnx_type) \
            case onnx_type:                              \
                ort_math_min_impl_##c_type(              \
                    result->data,                 \
                    tensor->data, element_count);        \
                break;
            ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_MIN_CASE)
            #undef ORT_MATH_MIN_CASE

            default:
                php_ort_status_throw(php_ort_status_math_invalidtype_ce,
                    "min: unsupported data type for reduction operation");
                return NULL;
        }

        return result;
    }

    // Handle specified axis
    if (Z_TYPE_P(axis_zval) != IS_LONG) {
        php_ort_status_throw(php_ort_status_tensor_invalidshape_ce, 
            "min: axis must be an integer");
        return NULL;
    }

    int64_t axis = Z_LVAL_P(axis_zval);
    if (!ort_math_validate_axis(tensor, axis, "min")) {
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
    if (result_dims == 0 && !keepdims) {
        result_dims = 1;
        result_shape[0] = 1;
    }

    ort_tensor_t* result = ort_math_result_tensor(result_shape, result_dims, tensor->type, "min_result");

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

    switch (tensor->type) {
        #define ORT_MATH_MIN_AXIS_CASE(c_type, onnx_type) \
        case onnx_type:                                   \
            ort_math_min_axis_impl_##c_type(              \
                result->data,                      \
                tensor->data,                             \
                outer_size, axis_size, inner_size);       \
            break;
        ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_MIN_AXIS_CASE)
        #undef ORT_MATH_MIN_AXIS_CASE

        default:
            php_ort_status_throw(php_ort_status_math_invalidtype_ce,
                "min: unsupported data type for reduction operation");
            efree(strides);
            efree(result_shape);
            return NULL;
    }

    efree(strides);
    efree(result_shape);

    return result;
}