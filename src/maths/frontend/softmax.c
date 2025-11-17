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
 @brief Implements frontend softmax operations for tensors
 @test tests/math/transform/softmax
*/

#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/result.h"
#include "maths/schema/softmax.h"

ORT_MATH_FRONTEND_TRANSFORMATION_AXIS_OP_DECL(softmax, float16) {
    float16* va = (float16*)a;
    float16* res = (float16*)result;
    size_t outer = 1, inner = 1;
    for (size_t i = 0; i < axis; ++i) outer *= shape[i];
    for (size_t i = axis + 1; i < dims; ++i) inner *= shape[i];
    
    for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        for (size_t inner_idx = 0; inner_idx < inner; ++inner_idx) {
            int64_t indices[ORT_MATH_RESULT_STACK_DIMENSIONS];
            for (size_t d = 0; d < dims; ++d) indices[d] = 0;
            
            size_t tmp_outer = outer_idx;
            for (size_t d = 0; d < axis; ++d) {
                indices[d] = tmp_outer % shape[d];
                tmp_outer /= shape[d];
            }
            size_t tmp_inner = inner_idx;
            for (size_t d = dims - 1; d > axis; --d) {
                indices[d] = tmp_inner % shape[d];
                tmp_inner /= shape[d];
            }
            
            // Find max for numerical stability
            float32 max = -INFINITY;
            for (size_t axis_idx = 0; axis_idx < shape[axis]; ++axis_idx) {
                indices[axis] = axis_idx;
                size_t flat = ort_math_result_flat(indices, shape, dims);
                float32 val = ort_math_float32_from_float16(va[flat]);
                if (val > max)
                    max = val;
            }
            
            // Compute exp(x - max) and sum
            float32 sum = 0;
            for (size_t axis_idx = 0; axis_idx < shape[axis]; ++axis_idx) {
                indices[axis] = axis_idx;
                size_t flat = ort_math_result_flat(indices, shape, dims);
                float32 exp_val = exp((float64)ort_math_float32_from_float16(va[flat]) - (float64)max);
                res[flat] = ort_math_float16_from_float32(exp_val);
                sum += exp_val;
            }
            
            // Normalize by sum
            for (size_t axis_idx = 0; axis_idx < shape[axis]; ++axis_idx) {
                indices[axis] = axis_idx;
                size_t flat = ort_math_result_flat(indices, shape, dims);
                float32 val = ort_math_float32_from_float16(res[flat]);
                res[flat] = ort_math_float16_from_float32(val / sum);
            }
        }
    }
}

#define ORT_MATH_FRONTEND_SOFTMAX_AXIS_IMPL_FOR_TYPE(c_type, unused) \
    ORT_MATH_FRONTEND_TRANSFORMATION_AXIS_OP_DECL(softmax, c_type) { \
        c_type* va = (c_type*)a; \
        c_type* res = (c_type*)result; \
        size_t outer = 1, inner = 1; \
        for (size_t i = 0; i < axis; ++i) outer *= shape[i]; \
        for (size_t i = axis + 1; i < dims; ++i) inner *= shape[i]; \
        for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) { \
            for (size_t inner_idx = 0; inner_idx < inner; ++inner_idx) { \
                int64_t indices[ORT_MATH_RESULT_STACK_DIMENSIONS]; \
                for (size_t d = 0; d < dims; ++d) indices[d] = 0; \
                size_t tmp_outer = outer_idx; \
                for (size_t d = 0; d < axis; ++d) { \
                    indices[d] = tmp_outer % shape[d]; \
                    tmp_outer /= shape[d]; \
                } \
                size_t tmp_inner = inner_idx; \
                for (size_t d = dims - 1; d > axis; --d) { \
                    indices[d] = tmp_inner % shape[d]; \
                    tmp_inner /= shape[d]; \
                } \
                c_type max = -INFINITY; \
                for (size_t axis_idx = 0; axis_idx < shape[axis]; ++axis_idx) { \
                    indices[axis] = axis_idx; \
                    size_t flat = ort_math_result_flat(indices, shape, dims); \
                    if (va[flat] > max) max = va[flat]; \
                } \
                c_type sum = 0; \
                for (size_t axis_idx = 0; axis_idx < shape[axis]; ++axis_idx) { \
                    indices[axis] = axis_idx; \
                    size_t flat = ort_math_result_flat(indices, shape, dims); \
                    res[flat] = (c_type)exp((float64)va[flat] - (float64)max); \
                    sum += res[flat]; \
                } \
                for (size_t axis_idx = 0; axis_idx < shape[axis]; ++axis_idx) { \
                    indices[axis] = axis_idx; \
                    size_t flat = ort_math_result_flat(indices, shape, dims); \
                    res[flat] /= sum; \
                } \
            } \
        } \
    }

ORT_MATH_FOREACH_REAL_TYPE(
    ORT_MATH_FRONTEND_SOFTMAX_AXIS_IMPL_FOR_TYPE)
#undef ORT_MATH_FRONTEND_SOFTMAX_AXIS_IMPL_FOR_TYPE

static ort_math_kernel_transform_axis_t
    ort_math_frontend_dispatch_transform_axis_softmax(
        ort_math_promotion_t *promotion,
        const ort_math_promotion_schema_t *schema) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(
            promotion->result_type);
    return dispatch->softmax_axis_func;
}

ORT_MATH_RESULT_TRANSFORM_AXIS_IMPL(softmax,
    ort_math_frontend_dispatch_transform_axis_softmax,
    ort_math_validate_input,
    ort_math_validate_axis,
    &ort_math_promotion_schema_softmax)