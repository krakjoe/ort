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
 @brief Implements frontend real operations for tensors
 @test tests/math/real
*/

#include "ort.h"

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"

#include "maths/schema/abs.h"
#include "maths/schema/acos.h"
#include "maths/schema/arccos.h"
#include "maths/schema/arccosh.h"
#include "maths/schema/arcsin.h"
#include "maths/schema/arcsinh.h"
#include "maths/schema/arctan.h"
#include "maths/schema/arctanh.h"
#include "maths/schema/asin.h"
#include "maths/schema/atan.h"
#include "maths/schema/cbrt.h"
#include "maths/schema/ceil.h"
#include "maths/schema/cos.h"
#include "maths/schema/cosh.h"
#include "maths/schema/exp.h"
#include "maths/schema/exp2.h"
#include "maths/schema/floor.h"
#include "maths/schema/log.h"
#include "maths/schema/log2.h"
#include "maths/schema/log10.h"
#include "maths/schema/round.h"
#include "maths/schema/sin.h"
#include "maths/schema/sinh.h"
#include "maths/schema/tan.h"
#include "maths/schema/tanh.h"
#include "maths/schema/trunc.h"

/* =============================================================================
 * REAL MATHEMATICAL FUNCTIONS
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

#define ORT_MATH_REAL_EXPORT_WITH_DISPATCH(func_name, math_func_f, math_func_d)       \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(float, func_name, math_func_f)                      \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(double, func_name, math_func_d)                     \
    ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(ort_math_unary_op_func_t, func_name)

#define ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(func_name, math_func_f, math_func_d) \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(float, func_name, math_func_f)                   \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(double, func_name, math_func_d)                  \
    ORT_MATH_FRONTEND_DISPATCH_DECL(ort_math_unary_op_func_t, func_name) {         \
    switch (promotion->result_type) {                                              \
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:                                  \
            return ORT_MATH_FRONTEND_OP_SYMBOL(func_name, float);                  \
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:                                 \
            return ORT_MATH_FRONTEND_OP_SYMBOL(func_name, double);                 \
        default: return NULL;                                                      \
    }                                                                              \
}

/* Generate all real mathematical functions */
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(sin, sinf, sin)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(cos, cosf, cos)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(tan, tanf, tan)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(arccos, acosf, acos)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(arcsin, asinf, asin)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(arctan, atanf, atan)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(arccosh, acoshf, acosh)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(arcsinh, asinhf, asinh)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(arctanh, atanhf, atanh)
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

/* {{{ */
#define ORT_MATH_REAL_EXPORT_ABS_FOR_INTEGERS(c_type, unused)       \
ORT_MATH_FRONTEND_UNARY_OP_DECL(abs, c_type) {                      \
    c_type* res = (c_type*)result;                                  \
    const c_type* va = (const c_type*)a;                            \
    for (size_t i = 0; i < count; i++) {                            \
        res[i] = va[i] < 0 ? -va[i] : va[i];                        \
    }                                                               \
}

ORT_MATH_FOREACH_INTEGER_TYPE(
    ORT_MATH_REAL_EXPORT_ABS_FOR_INTEGERS)

ORT_MATH_REAL_EXPORT_WITH_DISPATCH(abs, fabsf, fabs) /* }}} */

/* {{{ */
#define ORT_MATH_REAL_EXPORT_DUMMY_FOR_INTEGERS(c_type, unused, name)      \
ORT_MATH_FRONTEND_UNARY_OP_DECL(name, c_type) {                     \
    c_type* res = (c_type*)result;                                  \
    const c_type* va = (const c_type*)a;                            \
    for (size_t i = 0; i < count; i++) {                            \
        res[i] = va[i];                                             \
    }                                                               \
} /* }}} */

/* {{{ */
ORT_MATH_FOREACH_INTEGER_TYPE_EX(
    ORT_MATH_REAL_EXPORT_DUMMY_FOR_INTEGERS, ceil)

ORT_MATH_REAL_EXPORT_WITH_DISPATCH(ceil, ceilf, ceil) /* }}} */

/* {{{ */
ORT_MATH_FOREACH_INTEGER_TYPE_EX(
    ORT_MATH_REAL_EXPORT_DUMMY_FOR_INTEGERS, floor)

ORT_MATH_REAL_EXPORT_WITH_DISPATCH(floor, floorf, floor) /* }}} */

/* {{{ */
ORT_MATH_FOREACH_INTEGER_TYPE_EX(
    ORT_MATH_REAL_EXPORT_DUMMY_FOR_INTEGERS, round)

ORT_MATH_REAL_EXPORT_WITH_DISPATCH(round, roundf, round) /* }}} */

/* {{{  */
ORT_MATH_FOREACH_INTEGER_TYPE_EX(
    ORT_MATH_REAL_EXPORT_DUMMY_FOR_INTEGERS, trunc)

ORT_MATH_REAL_EXPORT_WITH_DISPATCH(trunc, truncf, trunc) /* }}} */

/* =============================================================================
 * PUBLIC INTERFACE FUNCTIONS FOR REAL MATHEMATICAL OPERATIONS
 * =============================================================================
 */

ORT_MATH_RESULT_UNARY_IMPL(sin, ort_math_frontend_dispatch_sin, &ort_math_promotion_schema_sin)
ORT_MATH_RESULT_UNARY_IMPL(cos, ort_math_frontend_dispatch_cos, &ort_math_promotion_schema_cos)
ORT_MATH_RESULT_UNARY_IMPL(tan, ort_math_frontend_dispatch_tan, &ort_math_promotion_schema_tan)
ORT_MATH_RESULT_UNARY_IMPL(asin, ort_math_frontend_dispatch_asin, &ort_math_promotion_schema_asin)
ORT_MATH_RESULT_UNARY_IMPL(acos, ort_math_frontend_dispatch_acos, &ort_math_promotion_schema_acos)
ORT_MATH_RESULT_UNARY_IMPL(atan, ort_math_frontend_dispatch_atan, &ort_math_promotion_schema_atan)
ORT_MATH_RESULT_UNARY_IMPL(sinh, ort_math_frontend_dispatch_sinh, &ort_math_promotion_schema_sinh)
ORT_MATH_RESULT_UNARY_IMPL(cosh, ort_math_frontend_dispatch_cosh, &ort_math_promotion_schema_cosh)
ORT_MATH_RESULT_UNARY_IMPL(tanh, ort_math_frontend_dispatch_tanh, &ort_math_promotion_schema_tanh)
ORT_MATH_RESULT_UNARY_IMPL(exp, ort_math_frontend_dispatch_exp, &ort_math_promotion_schema_exp)
ORT_MATH_RESULT_UNARY_IMPL(exp2, ort_math_frontend_dispatch_exp2, &ort_math_promotion_schema_exp2)
ORT_MATH_RESULT_UNARY_IMPL(log, ort_math_frontend_dispatch_log, &ort_math_promotion_schema_log)
ORT_MATH_RESULT_UNARY_IMPL(log2, ort_math_frontend_dispatch_log2, &ort_math_promotion_schema_log2)
ORT_MATH_RESULT_UNARY_IMPL(log10, ort_math_frontend_dispatch_log10, &ort_math_promotion_schema_log10)
ORT_MATH_RESULT_UNARY_IMPL(cbrt, ort_math_frontend_dispatch_cbrt, &ort_math_promotion_schema_cbrt)

ORT_MATH_RESULT_UNARY_IMPL(arccos, ort_math_frontend_dispatch_arccos, &ort_math_promotion_schema_arccos)
ORT_MATH_RESULT_UNARY_IMPL(arccosh, ort_math_frontend_dispatch_arccosh, &ort_math_promotion_schema_arccosh)
ORT_MATH_RESULT_UNARY_IMPL(arcsin, ort_math_frontend_dispatch_arcsin, &ort_math_promotion_schema_arcsin)
ORT_MATH_RESULT_UNARY_IMPL(arcsinh, ort_math_frontend_dispatch_arcsinh, &ort_math_promotion_schema_arcsinh)
ORT_MATH_RESULT_UNARY_IMPL(arctan, ort_math_frontend_dispatch_arctan, &ort_math_promotion_schema_arctan)
ORT_MATH_RESULT_UNARY_IMPL(arctanh, ort_math_frontend_dispatch_arctanh, &ort_math_promotion_schema_arctanh)

ORT_MATH_RESULT_UNARY_IMPL(abs, ort_math_frontend_dispatch_abs, &ort_math_promotion_schema_abs)
ORT_MATH_RESULT_UNARY_IMPL(ceil, ort_math_frontend_dispatch_ceil, &ort_math_promotion_schema_ceil)
ORT_MATH_RESULT_UNARY_IMPL(floor, ort_math_frontend_dispatch_floor,&ort_math_promotion_schema_floor)
ORT_MATH_RESULT_UNARY_IMPL(round, ort_math_frontend_dispatch_round, &ort_math_promotion_schema_round)
ORT_MATH_RESULT_UNARY_IMPL(trunc, ort_math_frontend_dispatch_trunc, &ort_math_promotion_schema_trunc)