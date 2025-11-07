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

#include "maths/backend/neon/impl.h"

#include <arm_neon.h>  /* NEON */

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, div, float16) {
    const float16* va = (const float16*)a;
    const float16* vb = (const float16*)b;
    float16* res      = (float16*)result;
    const size_t mw = 8; /* 8 float16 per NEON register */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_div_float16_fallback;
    }

    // Vectorized loop - process 8 float16 at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float16x8_t va8 = vld1q_f16(
            (const float16_t*)&va[i]);
        float16x8_t vb8 = vld1q_f16(
            (const float16_t*)&vb[i]);
        float16x8_t vr8 = vdivq_f16(va8, vb8);
        vst1q_f16((float16_t*)&res[i], vr8);
    }

__ort_math_backend_div_float16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(div, float16)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}
#endif

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, div, float32) {
    const float32* va = (const float32*)a;
    const float32* vb = (const float32*)b;
    float32* res      = (float32*)result;
    const size_t mw = 4; /* 4 float32 per NEON register */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_div_float32_fallback;
    }

    // Vectorized loop - process 4 float32 at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float32x4_t va4 = vld1q_f32(&va[i]);
        float32x4_t vb4 = vld1q_f32(&vb[i]);
        float32x4_t vr4 = vdivq_f32(va4, vb4);
        vst1q_f32(&res[i], vr4);
    }

__ort_math_backend_div_float32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(div, float32)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, div, float64) {
    const float64* va = (const float64*)a;
    const float64* vb = (const float64*)b;
    float64* res = (float64*)result;
    const size_t mw = 2; // 2 float64 per NEON register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_div_float64_fallback;
    }

    // Vectorized loop - process 2 float64 at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float64x2_t ma = vld1q_f64(&va[i]);
        float64x2_t mb = vld1q_f64(&vb[i]);
        float64x2_t mr = vdivq_f64(ma, mb);
        vst1q_f64(&res[i], mr);
    }

__ort_math_backend_div_float64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(div, float64)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}