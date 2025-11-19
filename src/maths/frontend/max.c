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
 @brief Implements frontend max operations for tensors
 @test tests/math/reduce/max
*/

#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/result.h"
#include "maths/schema/max.h"

ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(max, float16) {
    float16* va = (float16*)a;
    float16* res = (float16*)result;
    ORT_MATH_REDUCTION_KERNEL_AXIS_PROLOGUE()

    float32 max = ort_math_float32_from_float16(
        va[ort_math_result_flat(
            indices, input_shape, input_dims)]
    );
    for (size_t axis_idx = 1;
                axis_idx < input_shape[axis];
                ++axis_idx) {
        indices[axis] = axis_idx;
        size_t flat = ort_math_result_flat(
            indices, input_shape, input_dims);
        float32 val = ort_math_float32_from_float16(va[flat]);
        if (val > max)
            max = val;
    }

    ORT_MATH_REDUCTION_KERNEL_AXIS_EPILOGUE()

    size_t out_flat = ort_math_result_flat(
        out_indices, output_shape, output_dims);
    res[out_flat] = ort_math_float16_from_float32(max);

    ORT_MATH_REDUCTION_KERNEL_AXIS_LEAVE()
}

#define ORT_MATH_FRONTEND_MAX_AXIS_IMPL_FOR_TYPE(c_type, unused) \
    ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(max, c_type) { \
        c_type* va = (c_type*)a; \
        c_type* res = (c_type*)result; \
        ORT_MATH_REDUCTION_KERNEL_AXIS_PROLOGUE() \
        \
        c_type max = va[ort_math_result_flat(\
            indices, input_shape, input_dims)]; \
        for (size_t axis_idx = 1; \
                    axis_idx < input_shape[axis]; \
                    ++axis_idx) { \
            indices[axis] = axis_idx; \
            size_t flat = ort_math_result_flat(\
                indices, input_shape, input_dims); \
            if (va[flat] > max) max = va[flat]; \
        } \
        \
        ORT_MATH_REDUCTION_KERNEL_AXIS_EPILOGUE() \
        \
        size_t out_flat = ort_math_result_flat(\
            out_indices, output_shape, output_dims); \
        res[out_flat] = max; \
        \
        ORT_MATH_REDUCTION_KERNEL_AXIS_LEAVE() \
    }

ORT_MATH_FOREACH_NUMERIC_TYPE(
    ORT_MATH_FRONTEND_MAX_AXIS_IMPL_FOR_TYPE)
#undef ORT_MATH_FRONTEND_MAX_AXIS_IMPL_FOR_TYPE

ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(max, zend_bool) {
    zend_bool* va = (zend_bool*)a;
    zend_bool* res = (zend_bool*)result;
    ORT_MATH_REDUCTION_KERNEL_AXIS_PROLOGUE()

    zend_bool max = va[ort_math_result_flat(
        indices, input_shape, input_dims)];
    for (size_t axis_idx = 1;
                axis_idx < input_shape[axis];
                ++axis_idx) {
        indices[axis] = axis_idx;
        size_t flat = ort_math_result_flat(
            indices, input_shape, input_dims);
        max = max || va[flat];
    }

    ORT_MATH_REDUCTION_KERNEL_AXIS_EPILOGUE()   

    size_t out_flat = ort_math_result_flat(
        out_indices, output_shape, output_dims);
    res[out_flat] = max;

    ORT_MATH_REDUCTION_KERNEL_AXIS_LEAVE()  
}

ORT_MATH_FRONTEND_UNARY_OP_DECL(max, float16) {
    float16* va = (float16*)a;
    float16* res = (float16*)result;
    float32 max = ort_math_float32_from_float16(va[0]);
    for (size_t idx = 1; idx < count; idx++) {
        float32 val = ort_math_float32_from_float16(va[idx]);
        if (val > max)
            max = val;
    }
    res[0] = ort_math_float16_from_float32(max);
}

#define ORT_MATH_FRONTEND_MAX_IMPL_FOR_TYPE(c_type, unused) \
    ORT_MATH_FRONTEND_UNARY_OP_DECL(max, c_type) { \
        c_type* va = (c_type*)a;                   \
        c_type* res = (c_type*)result;             \
        c_type max = va[0];                        \
        for (size_t idx = 1; idx < count; idx++) { \
            if (va[idx] > max) max = va[idx];      \
        }                                          \
        res[0] = max;                              \
    }

ORT_MATH_FOREACH_NUMERIC_TYPE(
    ORT_MATH_FRONTEND_MAX_IMPL_FOR_TYPE)
#undef ORT_MATH_FRONTEND_MAX_IMPL_FOR_TYPE

ORT_MATH_FRONTEND_UNARY_OP_DECL(max, zend_bool) {
    zend_bool* va = (zend_bool*)a;
    zend_bool* res = (zend_bool*)result;
    zend_bool max = va[0];
    for (size_t idx = 1; idx < count; idx++) {
        max = max || va[idx];
    }
    res[0] = max;
}

static zend_always_inline ort_math_kernel_unary_t
    ort_math_frontend_dispatch_reduce_tensor_max(
        ort_math_promotion_t* promotion,
        const ort_math_promotion_schema_t* schema) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(
            promotion->result_type);
    return dispatch->max_func;
}

ORT_MATH_RESULT_REDUCE_TENSOR_IMPL(max,
    ort_math_frontend_dispatch_reduce_tensor_max,
    ort_math_validate_input,
    &ort_math_promotion_schema_max);

static ort_math_kernel_reduce_axis_t
    ort_math_frontend_dispatch_reduce_axis_max(
        ort_math_promotion_t* promotion,
        const ort_math_promotion_schema_t* schema) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(
            promotion->result_type);
    return dispatch->max_axis_func;
}

ORT_MATH_RESULT_REDUCE_AXIS_IMPL(max,
    ort_math_frontend_dispatch_reduce_axis_max,
    ort_math_validate_input,
    ort_math_validate_axis,
    ort_math_result_reduce,
    &ort_math_promotion_schema_max)
