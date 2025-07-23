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
 @brief Implements frontend sum operations for tensors
 @test tests/math/reduce/sum
*/

#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/result.h"
#include "maths/schema/sum.h"

#define ORT_MATH_FRONTEND_SUM_AXIS_IMPL_FOR_TYPE(c_type) \
    ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(sum, c_type) { \
        c_type* va = (c_type*)a; \
        c_type* res = (c_type*)result; \
        size_t outer = 1, inner = 1; \
        for (size_t i = 0; i < axis; ++i) outer *= input_shape[i]; \
        for (size_t i = axis + 1; i < input_dims; ++i) inner *= input_shape[i]; \
        for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) { \
            for (size_t inner_idx = 0; inner_idx < inner; ++inner_idx) { \
                int64_t indices[ORT_MATH_RESULT_STACK_DIMENSIONS]; \
                for (size_t d = 0; d < input_dims; ++d) indices[d] = 0; \
                size_t tmp_outer = outer_idx; \
                for (size_t d = 0; d < axis; ++d) { \
                    indices[d] = tmp_outer % input_shape[d]; \
                    tmp_outer /= input_shape[d]; \
                } \
                size_t tmp_inner = inner_idx; \
                for (size_t d = input_dims - 1; d > axis; --d) { \
                    indices[d] = tmp_inner % input_shape[d]; \
                    tmp_inner /= input_shape[d]; \
                } \
                c_type sum = 0; \
                for (size_t axis_idx = 0; axis_idx < input_shape[axis]; ++axis_idx) { \
                    indices[axis] = axis_idx; \
                    size_t flat = ort_math_result_flat(indices, input_shape, input_dims); \
                    sum += va[flat]; \
                } \
                /* Write output using output_shape/output_dims */ \
                int64_t out_indices[ORT_MATH_RESULT_STACK_DIMENSIONS]; \
                if (output_dims == input_dims) { /* keepdims=true */ \
                    for (size_t d = 0; d < output_dims; ++d) { \
                        if (d == axis) { \
                            out_indices[d] = 0; \
                        } else { \
                            out_indices[d] = indices[d]; \
                        } \
                    } \
                } else { /* keepdims=false */ \
                    size_t out_j = 0; \
                    for (size_t d = 0; d < input_dims; ++d) { \
                        if (d != axis) { \
                            out_indices[out_j++] = indices[d]; \
                        } \
                    } \
                } \
                size_t out_flat = ort_math_result_flat(out_indices, output_shape, output_dims); \
                res[out_flat] = sum; \
            } \
        } \
    }

ORT_MATH_FRONTEND_SUM_AXIS_IMPL_FOR_TYPE(float)
ORT_MATH_FRONTEND_SUM_AXIS_IMPL_FOR_TYPE(double)
ORT_MATH_FRONTEND_SUM_AXIS_IMPL_FOR_TYPE(int64_t)
#undef ORT_MATH_FRONTEND_SUM_AXIS_IMPL_FOR_TYPE

#define ORT_MATH_FRONTEND_SUM_IMPL_FOR_TYPE(c_type)    \
    ORT_MATH_FRONTEND_REDUCTION_OP_DECL(sum, c_type) { \
        c_type* va = (c_type*)a;                       \
        c_type* res = (c_type*)result;                 \
        c_type sum = 0;                                \
        for (size_t idx = 0; idx < count; idx++) {     \
            sum += va[idx];                            \
        }                                              \
        res[0] = sum;                                  \
    }

ORT_MATH_FRONTEND_SUM_IMPL_FOR_TYPE(float)
ORT_MATH_FRONTEND_SUM_IMPL_FOR_TYPE(double)
ORT_MATH_FRONTEND_SUM_IMPL_FOR_TYPE(int64_t)
#undef ORT_MATH_FRONTEND_SUM_IMPL_FOR_TYPE

static ort_math_kernel_unary_t
    ort_math_frontend_dispatch_reduce_tensor_sum(
        ort_math_promotion_t *promotion,
        const ort_math_promotion_schema_t *schema) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(
            promotion->result_type);
    return dispatch->sum_func;
}

ORT_MATH_RESULT_REDUCE_TENSOR_IMPL(sum,
    ort_math_frontend_dispatch_reduce_tensor_sum,
    ort_math_validate_input,
    &ort_math_promotion_schema_sum);

static ort_math_kernel_reduce_axis_t
    ort_math_frontend_dispatch_reduce_axis_sum(
        ort_math_promotion_t *promotion,
        const ort_math_promotion_schema_t *schema) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(
            promotion->result_type);
    return dispatch->sum_axis_func;
}

ORT_MATH_RESULT_REDUCE_AXIS_IMPL(sum,
    ort_math_frontend_dispatch_reduce_axis_sum,
    ort_math_validate_input,
    ort_math_validate_axis,
    ort_math_result_reduce,
    &ort_math_promotion_schema_sum)