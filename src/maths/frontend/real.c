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

#include "ort.h"

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"

/* =============================================================================
 * UNARY MATHEMATICAL FUNCTIONS
 * =============================================================================
 */

#define ORT_MATH_REAL_EXPORT_FOR_TYPE(c_type, func_name, math_func) \
ORT_MATH_FRONTEND_UNARY_OP_DECL(func_name, c_type) {                \
    c_type* res = (c_type*)result;                                  \
    const c_type* va = (const c_type*)a;                            \
    for (size_t i = 0; i < count; i++) {                            \
        res[i] = math_func(va[i]);                                  \
    }                                                               \
}

#define ORT_MATH_REAL_EXPORT_WITH_DISPATCH(func_name, math_func_f, math_func_d) \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(float, func_name, math_func_f)         \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(double, func_name, math_func_d)        \
    static ort_math_unary_op_func_t ort_math_frontend_get_##func_name##_func( \
        ONNXTensorElementDataType type) {                                \
    const ort_math_dispatch_t* dispatch =                                \
        ort_math_dispatch_type(type);                                    \
    return dispatch->func_name##_func;                                   \
}

#define ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(func_name, math_func_f, math_func_d) \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(float, func_name, math_func_f)                   \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(double, func_name, math_func_d)                  \
    static ort_math_unary_op_func_t ort_math_frontend_get_##func_name##_func(      \
        ONNXTensorElementDataType type) {                                          \
    switch (type) {                                                                \
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:                                  \
            return ORT_MATH_FRONTEND_OP_SYMBOL(func_name, float);                  \
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:                                 \
            return ORT_MATH_FRONTEND_OP_SYMBOL(func_name, double);                 \
        default: return NULL;                                                      \
    }                                                                              \
}

/* Generate all mathematical functions */
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(sin, sinf, sin)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(cos, cosf, cos)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(tan, tanf, tan)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(asin, asinf, asin)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(acos, acosf, acos)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(atan, atanf, atan)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(sinh, sinhf, sinh)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(cosh, coshf, cosh)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(tanh, tanhf, tanh)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(exp, expf, exp)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(exp2, exp2f, exp2)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(log, logf, log)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(log2, log2f, log2)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(log10, log10f, log10)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(cbrt, cbrtf, cbrt)

ORT_MATH_REAL_EXPORT_WITH_DISPATCH(abs, fabsf, fabs)
ORT_MATH_REAL_EXPORT_WITH_DISPATCH(ceil, ceilf, ceil)
ORT_MATH_REAL_EXPORT_WITH_DISPATCH(floor, floorf, floor)
ORT_MATH_REAL_EXPORT_WITH_DISPATCH(round, roundf, round)
ORT_MATH_REAL_EXPORT_WITH_DISPATCH(trunc, truncf, trunc)

/* =============================================================================
 * PUBLIC INTERFACE FUNCTIONS FOR UNARY MATHEMATICAL OPERATIONS
 * =============================================================================
 */

static zend_always_inline ONNXTensorElementDataType ort_math_frontend_real_get_promotion_schema(ONNXTensorElementDataType type) {
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        return type;
    }

    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
}

ORT_MATH_UNARY_RESULT_IMPL(sin, ort_math_frontend_get_sin_func)
ORT_MATH_UNARY_RESULT_IMPL(cos, ort_math_frontend_get_cos_func)
ORT_MATH_UNARY_RESULT_IMPL(tan, ort_math_frontend_get_tan_func)
ORT_MATH_UNARY_RESULT_IMPL(asin, ort_math_frontend_get_asin_func)
ORT_MATH_UNARY_RESULT_IMPL(acos, ort_math_frontend_get_acos_func)
ORT_MATH_UNARY_RESULT_IMPL(atan, ort_math_frontend_get_atan_func)
ORT_MATH_UNARY_RESULT_IMPL(sinh, ort_math_frontend_get_sinh_func)
ORT_MATH_UNARY_RESULT_IMPL(cosh, ort_math_frontend_get_cosh_func)
ORT_MATH_UNARY_RESULT_IMPL(tanh, ort_math_frontend_get_tanh_func)
ORT_MATH_UNARY_RESULT_IMPL(exp, ort_math_frontend_get_exp_func)
ORT_MATH_UNARY_RESULT_IMPL(exp2, ort_math_frontend_get_exp2_func)
ORT_MATH_UNARY_PROMOTE_RESULT_IMPL(log, ort_math_frontend_get_log_func, ort_math_frontend_real_get_promotion_schema)
ORT_MATH_UNARY_PROMOTE_RESULT_IMPL(log2, ort_math_frontend_get_log2_func, ort_math_frontend_real_get_promotion_schema)
ORT_MATH_UNARY_PROMOTE_RESULT_IMPL(log10, ort_math_frontend_get_log10_func, ort_math_frontend_real_get_promotion_schema)
ORT_MATH_UNARY_RESULT_IMPL(ceil, ort_math_frontend_get_ceil_func)
ORT_MATH_UNARY_RESULT_IMPL(floor, ort_math_frontend_get_floor_func)
ORT_MATH_UNARY_RESULT_IMPL(trunc, ort_math_frontend_get_trunc_func)
ORT_MATH_UNARY_RESULT_IMPL(round, ort_math_frontend_get_round_func)
ORT_MATH_UNARY_RESULT_IMPL(cbrt, ort_math_frontend_get_cbrt_func)
ORT_MATH_UNARY_RESULT_IMPL(abs, ort_math_frontend_get_abs_func)
