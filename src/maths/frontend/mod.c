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
 @brief Implements frontend modulo operations for tensors
 @test tests/math/mod
*/
#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/schema/mod.h"

#define ORT_MATH_FRONTEND_MOD_IMPL_FOR_TYPE(c_type, unused)      \
    static zend_always_inline c_type ort_math_mod_impl_##c_type( \
        c_type a, c_type b) {                                    \
        if (b == 0) return 0;                                    \
        c_type r = a % b;                                        \
        if ((r != 0) && ((b < 0) != (r < 0)))                    \
            r += b;                                              \
        return r;                                                \
    }

ORT_MATH_FOREACH_INTEGER_TYPE(
    ORT_MATH_FRONTEND_MOD_IMPL_FOR_TYPE)
#undef ORT_MATH_FRONTEND_MOD_IMPL_FOR_TYPE

static zend_always_inline float ort_math_mod_impl_float(float a, float b) {
    float r = fmodf(a, b);
    return 
        (r < 0) ? r + b : r;
}

static zend_always_inline double ort_math_mod_impl_double(double a, double b) {
    double r = fmod(a, b);
    return 
        (r < 0) ? r + b : r;
}

#define ORT_MATH_FRONTEND_MOD_IMPL(c_type, onnx_type)               \
    ORT_MATH_FRONTEND_BINARY_OP_DECL(mod, c_type) {                 \
    c_type* res = (c_type*)result;                                  \
    const c_type* va = (const c_type*)a;                            \
    const c_type* vb = (const c_type*)b;                            \
    for (size_t i = 0; i < count; i++) {                            \
        res[i] = ort_math_mod_impl_##c_type(                        \
            va[i], vb[i]);                                          \
    }                                                               \
}

ORT_MATH_FOREACH_NUMERIC_TYPE(
    ORT_MATH_FRONTEND_MOD_IMPL)
#undef ORT_MATH_FRONTEND_MOD_IMPL

#define ORT_MATH_FRONTEND_MOD_SCALAR_IMPL(c_type, onnx_type)                 \
    ORT_MATH_FRONTEND_SCALAR_OP_DECL(mod, c_type) {                 \
    c_type* res = (c_type*)result;                                  \
    const c_type* va = (const c_type*)a;                            \
    const c_type sb = *(const c_type*)b;                            \
    for (size_t i = 0; i < count; i++) {                            \
        res[i] = ort_math_mod_impl_##c_type(va[i], sb);             \
    }                                                               \
}

ORT_MATH_FOREACH_NUMERIC_TYPE(
    ORT_MATH_FRONTEND_MOD_SCALAR_IMPL)
#undef ORT_MATH_FRONTEND_MOD_SCALAR_IMPL

ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(
    ort_math_kernel_binary_t, mod)
ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(
    ort_math_kernel_scalar_t, mod_scalar)

ORT_MATH_RESULT_BINARY_IMPL(mod,
    ORT_MATH_FRONTEND_DISPATCH_SYMBOL(mod),
    &ort_math_promotion_schema_mod)

ORT_MATH_RESULT_SCALAR_IMPL(mod,
    ORT_MATH_FRONTEND_DISPATCH_SYMBOL(mod_scalar),
    &ort_math_promotion_schema_mod)