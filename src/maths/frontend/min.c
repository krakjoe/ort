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
 @brief Implements frontend min operations for tensors
 @test tests/math/reduce/min
*/

#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/result.h"
#include "maths/schema/min.h"

#define ORT_MATH_MIN_AXIS_IMPL_FOR_TYPE(c_type, unused) \
    ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(min, c_type) { \
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

ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(min, zend_bool) {
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
    ORT_MATH_FRONTEND_REDUCTION_OP_DECL(min, c_type) { \
        c_type* va = (c_type*)a; \
        c_type* res = (c_type*)result; \
        c_type min = va[0]; \
        for (size_t idx = 1; idx < count; idx++) { \
            if (va[idx] < min) min = va[idx]; \
        } \
        res[0] = min; \
    }

ORT_MATH_FRONTEND_REDUCTION_OP_DECL(min, zend_bool) {
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

static ort_math_unary_op_func_t
    ort_math_frontend_get_reduce_tensor_min(
        ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->min_func;
}

ORT_MATH_RESULT_REDUCE_TENSOR_IMPL(min,
    ort_math_frontend_get_reduce_tensor_min,
    ort_math_validate_input,
    &ort_math_promotion_schema_min)

static ort_math_reduction_op_func_t
    ort_math_frontend_get_reduce_axis_min(
        ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->min_axis_func;
}

ORT_MATH_RESULT_REDUCE_AXIS_IMPL(min,
    ort_math_frontend_get_reduce_axis_min,
    ort_math_validate_input,
    ort_math_validate_axis,
    ort_math_result_reduce,
    &ort_math_promotion_schema_min);