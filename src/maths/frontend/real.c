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

#define ORT_MATH_REAL_EXPORT_WITH_DISPATCH(func_name, math_func_float16, math_func_float32, math_func_float64) \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(float16, func_name, math_func_float16)    \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(float32, func_name, math_func_float32)    \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(float64, func_name, math_func_float64)    \
    ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(ort_math_kernel_unary_t, func_name)

#define ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(func_name, math_func_float16, math_func_float32, math_func_float64) \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(float16, func_name, math_func_float16)           \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(float32, func_name, math_func_float32)           \
    ORT_MATH_REAL_EXPORT_FOR_TYPE(float64, func_name, math_func_float64)           \
    ORT_MATH_FRONTEND_DISPATCH_DECL(ort_math_kernel_unary_t, func_name) {          \
    switch (promotion->result_type) {                                              \
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:                                \
            return ORT_MATH_FRONTEND_OP_SYMBOL(func_name, float16);                \
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32:                                \
            return ORT_MATH_FRONTEND_OP_SYMBOL(func_name, float32);                \
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64:                                \
            return ORT_MATH_FRONTEND_OP_SYMBOL(func_name, float64);                \
        default: return NULL;                                                      \
    }                                                                              \
}

/* {{{ */
float16 sin16(float16 value) {
    return ort_math_float16_from_float32(
        sinf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(sin, sin16, sinf, sin) /* }}} */

/* {{{ */
float16 cos16(float16 value) {
    return ort_math_float16_from_float32(
        cosf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(cos, cos16, cosf, cos) /* }}} */

/* {{{ */
float16 tan16(float16 value) {
    return ort_math_float16_from_float32(
        tanf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(tan, tan16, tanf, tan) /* }}} */

/* {{{ */
float16 acosh16(float16 value) {
    return ort_math_float16_from_float32(
        acoshf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(arccosh, acosh16, acoshf, acosh) /* }}} */

/* {{{ */
float16 asinh16(float16 value) {
    return ort_math_float16_from_float32(
        asinhf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(arcsinh, asinh16, asinhf, asinh) /* }}} */

/* {{{ */
float16 atanh16(float16 value) {
    return ort_math_float16_from_float32(
        atanhf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(arctanh, atanh16, atanhf, atanh) /* }}} */

/* {{{ */
float16 asin16(float16 value) {
    return ort_math_float16_from_float32(
        asinf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(asin, asin16, asinf, asin) /* }}} */

/* {{{ */
float16 acos16(float16 value) {
    return ort_math_float16_from_float32(
        acosf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(acos, acos16, acosf, acos) /* }}} */

/* {{{ */
float16 atan16(float16 value) {
    return ort_math_float16_from_float32(
        atanf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(atan, atan16, atanf, atan) /* }}} */

/* {{{ */
float16 sinh16(float16 value) {
    return ort_math_float16_from_float32(
        sinhf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(sinh, sinh16, sinhf, sinh) /* }}} */

/* {{{ */
float16 cosh16(float16 value) {
    return ort_math_float16_from_float32(
        coshf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(cosh, cosh16, coshf, cosh) /* }}} */

/* {{{ */
float16 tanh16(float16 value) {
    return ort_math_float16_from_float32(
        tanhf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(tanh, tanh16, tanhf, tanh) /* }}} */

/* {{{ */
float16 exp16(float16 value) {
    return ort_math_float16_from_float32(
        expf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(exp, exp16, expf, exp) /* }}} */

/* {{{ */
float16 exp216(float16 value) {
    return ort_math_float16_from_float32(
        exp2f(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(exp2, exp216, exp2f, exp2) /* }}} */

/* {{{ */
float16 log16(float16 value) {
    return ort_math_float16_from_float32(
        logf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(log, log16, logf, log) /* }}} */

/* {{{ */
float16 log216(float16 value) {
    return ort_math_float16_from_float32(
        log2f(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(log2, log216, log2f, log2) /* }}} */

/* {{{ */
float16 log1016(float16 value) {
    return ort_math_float16_from_float32(
        log10f(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(log10, log1016, log10f, log10) /* }}} */

/* {{{ */
float16 cbrt16(float16 value) {
    return ort_math_float16_from_float32(
        cbrtf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(cbrt, cbrt16, cbrtf, cbrt) /* }}} */

/* {{{ */
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(arccos, acos16, acosf, acos)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(arcsin, asin16, asinf, asin)
ORT_MATH_REAL_EXPORT_WITHOUT_DISPATCH(arctan, atan16, atanf, atan) /* }}} */

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

float16 fabs16(float16 value) {
    return ort_math_float16_from_float32(
        fabsf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITH_DISPATCH(abs, fabs16, fabsf, fabs) /* }}} */

/* {{{ */
#define ORT_MATH_REAL_EXPORT_DUMMY_FOR_INTEGERS(c_type, unused, name) \
ORT_MATH_FRONTEND_UNARY_OP_DECL(name, c_type) {                       \
    c_type* res = (c_type*)result;                                    \
    const c_type* va = (const c_type*)a;                              \
    for (size_t i = 0; i < count; i++) {                              \
        res[i] = va[i];                                               \
    }                                                                 \
} /* }}} */

/* {{{ */
ORT_MATH_FOREACH_INTEGER_TYPE_EX(
    ORT_MATH_REAL_EXPORT_DUMMY_FOR_INTEGERS, ceil)

float16 ceil16(float16 value) {
    return ort_math_float16_from_float32(
        ceilf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITH_DISPATCH(ceil, ceil16, ceilf, ceil) /* }}} */

/* {{{ */
ORT_MATH_FOREACH_INTEGER_TYPE_EX(
    ORT_MATH_REAL_EXPORT_DUMMY_FOR_INTEGERS, floor)

float16 floor16(float16 value) {
    return ort_math_float16_from_float32(
        floorf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITH_DISPATCH(floor, floor16, floorf, floor) /* }}} */

/* {{{ */
ORT_MATH_FOREACH_INTEGER_TYPE_EX(
    ORT_MATH_REAL_EXPORT_DUMMY_FOR_INTEGERS, round)

float16 round16(float16 value) {
    return ort_math_float16_from_float32(
        roundf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITH_DISPATCH(round, round16, roundf, round) /* }}} */

/* {{{  */
ORT_MATH_FOREACH_INTEGER_TYPE_EX(
    ORT_MATH_REAL_EXPORT_DUMMY_FOR_INTEGERS, trunc)

float16 trunc16(float16 value) {
    return ort_math_float16_from_float32(
        truncf(
            ort_math_float32_from_float16(value)));
}

ORT_MATH_REAL_EXPORT_WITH_DISPATCH(trunc, trunc16, truncf, trunc) /* }}} */

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