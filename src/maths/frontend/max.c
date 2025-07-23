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

#define ORT_MATH_FRONTEND_MAX_AXIS_IMPL_FOR_TYPE(c_type, unused) \
    ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(max, c_type) { \
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

ORT_MATH_FOREACH_NUMERIC_TYPE(
    ORT_MATH_FRONTEND_MAX_AXIS_IMPL_FOR_TYPE)
#undef ORT_MATH_FRONTEND_MAX_AXIS_IMPL_FOR_TYPE

ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(max, zend_bool) {
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
    zend_bool max = 0;
    for (size_t idx = 0; idx < count; idx++) {
        max = max || va[idx];
    }
    res[0] = max;
}

static zend_always_inline ort_math_unary_op_func_t
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

static ort_math_reduction_op_func_t
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
