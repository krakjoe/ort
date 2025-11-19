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
 @brief Implements frontend argmax operations for tensors
 @test tests/math/argmax
*/

#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/result.h"
#include "maths/schema/argmax.h"

ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(argmax, float16) {
    float16* va = (float16*)a;
    int64_t* res = (int64_t*)result;
    ORT_MATH_REDUCTION_KERNEL_AXIS_PROLOGUE()

    indices[axis] = 0;
    size_t flat0 = ort_math_result_flat(
        indices, input_shape, input_dims);
    float32 max = ort_math_float32_from_float16(va[flat0]);
    int64_t argmax = 0;
    for (size_t axis_idx = 1;
                axis_idx < input_shape[axis];
                ++axis_idx) {
        indices[axis] = axis_idx;
        size_t flat = ort_math_result_flat(
            indices, input_shape, input_dims);
        float32 val = ort_math_float32_from_float16(va[flat]);
        if (val > max) {
            max = val;
            argmax = axis_idx;
        }
    }

    ORT_MATH_REDUCTION_KERNEL_AXIS_EPILOGUE()

    size_t out_flat = ort_math_result_flat(
        out_indices, output_shape, output_dims);
    res[out_flat] = argmax;

    ORT_MATH_REDUCTION_KERNEL_AXIS_LEAVE()
}

#define ORT_MATH_ARGMAX_AXIS_IMPL_FOR_TYPE(c_type, unused) \
    ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(argmax, c_type) { \
        c_type* va = (c_type*)a; \
        int64_t* res = (int64_t*)result; \
        ORT_MATH_REDUCTION_KERNEL_AXIS_PROLOGUE() \
        \
        indices[axis] = 0; \
        size_t flat0 = ort_math_result_flat( \
            indices, input_shape, input_dims); \
        c_type max = va[flat0]; \
        int64_t argmax = 0; \
        for (size_t axis_idx = 1; \
                    axis_idx < input_shape[axis]; \
                    ++axis_idx) { \
            indices[axis] = axis_idx; \
            size_t flat = ort_math_result_flat(\
                indices, input_shape, input_dims); \
            if (va[flat] > max) { \
                max = va[flat]; \
                argmax = axis_idx; \
            } \
        } \
        \
        ORT_MATH_REDUCTION_KERNEL_AXIS_EPILOGUE() \
        \
        size_t out_flat = ort_math_result_flat( \
            out_indices, output_shape, output_dims); \
        res[out_flat] = argmax; \
        \
        ORT_MATH_REDUCTION_KERNEL_AXIS_LEAVE() \
    }

ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(argmax, zend_bool) {
    zend_bool* va = (zend_bool*)a;
    int64_t* res = (int64_t*)result;
    ORT_MATH_REDUCTION_KERNEL_AXIS_PROLOGUE()

    int64_t argmax = -1;
    for (size_t axis_idx = 0;
                axis_idx < input_shape[axis];
                ++axis_idx) {
        indices[axis] = axis_idx;
        size_t flat = ort_math_result_flat(
            indices, input_shape, input_dims);
        if (va[flat]) {
            argmax = (int64_t)axis_idx;
            break;
        }
    }

    ORT_MATH_REDUCTION_KERNEL_AXIS_EPILOGUE()

    size_t out_flat = ort_math_result_flat(
        out_indices, output_shape, output_dims);
    res[out_flat] = argmax;

    ORT_MATH_REDUCTION_KERNEL_AXIS_LEAVE()
}

ORT_MATH_FRONTEND_REDUCTION_OP_DECL(argmax, float16) {
    float16* va = (float16*)a;
    int64_t* res = (int64_t*)result;
    float32 max = ort_math_float32_from_float16(va[0]);
    size_t argmax = 0;
    for (size_t idx = 1; idx < count; idx++) {
        float32 val = ort_math_float32_from_float16(va[idx]);
        if (max < val) {
            max = val;
            argmax = idx;
        }
    }
    res[0] = argmax;
}

#define ORT_MATH_ARGMAX_IMPL_FOR_TYPE(c_type, unused) \
    ORT_MATH_FRONTEND_REDUCTION_OP_DECL(argmax, c_type) { \
        c_type* va = (c_type*)a; \
        int64_t* res = (int64_t*)result; \
        c_type max = va[0]; \
        size_t argmax = 0; \
        for (size_t idx = 1; idx < count; idx++) { \
            c_type val = va[idx]; \
            if (max < val) { \
                max = val; \
                argmax = idx; \
            } \
        } \
        res[0] = argmax; \
    }

ORT_MATH_FRONTEND_REDUCTION_OP_DECL(argmax, zend_bool) {
    zend_bool* va = (zend_bool*)a;
    int64_t* res  = (int64_t*)result;
    for (size_t idx = 0; idx < count; idx++) {
        if (va[idx]) {
            res[0] = (int64_t) idx; // Return the index of the first true value
            return;                  // Stop at the first true value
        }
    }
    res[0] = -1; // If no true value found, return -1
}

ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_ARGMAX_AXIS_IMPL_FOR_TYPE)
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_ARGMAX_IMPL_FOR_TYPE)

static zend_always_inline int64_t* ort_math_frontend_reduce_shape_argmax(
    ort_tensor_t* tensor,
    size_t axis, zend_bool keepdims, size_t* result_dims) {
    size_t out_dims = keepdims ? tensor->dimensions : tensor->dimensions - 1;
    int64_t* result_shape = ecalloc(out_dims, sizeof(int64_t));
    size_t j = 0;
    for (size_t i = 0; i < tensor->dimensions; i++) {
        if (i == axis) {
            if (keepdims) {
                result_shape[j++] = 1;
            }
        } else {
            result_shape[j++] = tensor->shape[i];
        }
    }

    *result_dims = out_dims;
    return result_shape;
}

static ort_math_kernel_unary_t
    ort_math_frontend_dispatch_reduce_tensor_argmax(
        ort_math_promotion_t* promotion,
        const ort_math_promotion_schema_t* schema) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(
            promotion->inputs.lhs->type);
    return dispatch->argmax_func;
}

ORT_MATH_RESULT_SERIAL_REDUCE_TENSOR_IMPL(argmax,
    ort_math_frontend_dispatch_reduce_tensor_argmax,
    ort_math_validate_input,
    &ort_math_promotion_schema_argmax)

static ort_math_kernel_reduce_axis_t
    ort_math_frontend_dispatch_reduce_axis_argmax(
        ort_math_promotion_t* promotion,
        const ort_math_promotion_schema_t* schema) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(
            promotion->inputs.lhs->type);
    return dispatch->argmax_axis_func;
}

ORT_MATH_RESULT_SERIAL_REDUCE_AXIS_IMPL(argmax,
    ort_math_frontend_dispatch_reduce_axis_argmax,
    ort_math_validate_input,
    ort_math_validate_axis,
    ort_math_frontend_reduce_shape_argmax,
    &ort_math_promotion_schema_argmax);