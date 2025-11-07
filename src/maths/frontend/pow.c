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
 @brief Implements frontend power operations for tensors
 @test tests/math/pow
*/
#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/schema/pow.h"

#define ORT_MATH_FRONTEND_POW_IMPL_FOR_TYPE(c_type, unused)      \
    static zend_always_inline c_type ort_math_pow_impl_##c_type( \
        c_type a, c_type b) {                                    \
    c_type result =    1;                                        \
    if (b < 0) {                                                 \
        return 0;                                                \
    }                                                            \
    while (b) {                                                  \
        if (b & 1) result *= a;                                  \
        a *= a;                                                  \
        b >>= 1;                                                 \
    }                                                            \
    return result;                                               \
}

ORT_MATH_FOREACH_INTEGER_TYPE(
    ORT_MATH_FRONTEND_POW_IMPL_FOR_TYPE)
#undef ORT_MATH_FRONTEND_POW_IMPL_FOR_TYPE

#define ORT_MATH_FRONTEND_POW_IMPL(c_type, onnx_type)               \
    ORT_MATH_FRONTEND_BINARY_OP_DECL(pow, c_type) {                 \
    c_type* res = (c_type*)result;                                  \
    const c_type* va = (const c_type*)a;                            \
    const c_type* vb = (const c_type*)b;                            \
    for (size_t i = 0; i < count; i++) {                            \
        res[i] = ort_math_pow_impl_##c_type(                        \
            va[i], vb[i]);                                          \
    }                                                               \
}

ORT_MATH_FOREACH_INTEGER_TYPE(
    ORT_MATH_FRONTEND_POW_IMPL)
#undef ORT_MATH_FRONTEND_POW_IMPL

ORT_MATH_FRONTEND_BINARY_OP_DECL(pow, float16) {
    float16* res = (float16*)result;
    const float16* va = (const float16*)a;
    const float16* vb = (const float16*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = ort_math_float16_from_float32(
            powf(
                ort_math_float32_from_float16(va[i]),
                ort_math_float32_from_float16(vb[i]))
        );
    }
}

ORT_MATH_FRONTEND_BINARY_OP_DECL(pow, float32) {
    float32* res = (float32*)result;
    const float32* va = (const float32*)a;
    const float32* vb = (const float32*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = powf(va[i], vb[i]);
    }
}

ORT_MATH_FRONTEND_BINARY_OP_DECL(pow, float64) {
    float64* res = (float64*)result;
    const float64* va = (const float64*)a;
    const float64* vb = (const float64*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = pow(va[i], vb[i]);
    }
}

#define ORT_MATH_FRONTEND_POW_SCALAR_IMPL(c_type, onnx_type)          \
void ort_math_frontend_pow_scalar_##c_type(                           \
    void* result, const void* a, const void* b, size_t count) {       \
    c_type* res = (c_type*)result;                                    \
    const c_type* va = (const c_type*)a;                              \
    c_type sb = *(const c_type*)b;                                    \
    for (size_t i = 0; i < count; i++) {                              \
        res[i] = ort_math_pow_impl_##c_type(va[i], sb);               \
    }                                                                 \
}

ORT_MATH_FOREACH_INTEGER_TYPE(
    ORT_MATH_FRONTEND_POW_SCALAR_IMPL)
#undef ORT_MATH_FRONTEND_POW_SCALAR_IMPL

ORT_MATH_FRONTEND_SCALAR_OP_DECL(pow, float16) {
    float16* res = (float16*)result;
    const float16* va = (const float16*)a;
    float16 sb = *(const float16*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = ort_math_float16_from_float32(
            powf(
                ort_math_float32_from_float16(va[i]),
                ort_math_float32_from_float16(sb))
        );
    }
}

ORT_MATH_FRONTEND_SCALAR_OP_DECL(pow, float32) {
    float32* res = (float32*)result;
    const float32* va = (const float32*)a;
    float32 sb = *(const float32*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = powf(va[i], sb);
    }
}

ORT_MATH_FRONTEND_SCALAR_OP_DECL(pow, float64) {
    float64* res = (float64*)result;
    const float64* va = (const float64*)a;
    float64 sb = *(const float64*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = pow(va[i], sb);
    }
}

ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(
    ort_math_kernel_binary_t, pow)
ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(
    ort_math_kernel_scalar_t, pow_scalar)

ORT_MATH_RESULT_BINARY_IMPL(pow,
    ORT_MATH_FRONTEND_DISPATCH_SYMBOL(pow),
    &ort_math_promotion_schema_pow)
ORT_MATH_RESULT_SCALAR_IMPL(pow,
    ORT_MATH_FRONTEND_DISPATCH_SYMBOL(pow_scalar),
    &ort_math_promotion_schema_pow)