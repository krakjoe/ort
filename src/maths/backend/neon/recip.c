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
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, recip, float16) {
    const float16* va = (const float16*)a;
    float16* res      = (float16*)result;
    const size_t mw = 8; // 8 float16 per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_recip_float16_fallback;
    }

    // Vectorized loop - process 8 float16 at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float16x8_t ma = vld1q_f16(
            (const float16_t*)&va[i]);
        float16x8_t one = vdupq_n_f16(1.0f);
        float16x8_t mr = vdivq_f16(one, ma);  // 1.0f / x
        vst1q_f16((float16_t*)&res[i], mr);
    }

__ort_math_backend_recip_float16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(recip, float16)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
#endif

ORT_MATH_BACKEND_UNARY_OP_DECL(neon, recip, float32) {
    const float32* va = (const float32*)a;
    float32* res      = (float32*)result;
    const size_t mw = 4; // 4 float32 per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_recip_float32_fallback;
    }

    // Vectorized loop - process 4 float32 at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float32x4_t ma = vld1q_f32(&va[i]);
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t mr = vdivq_f32(one, ma);  // 1.0f / x
        vst1q_f32(&res[i], mr);
    }

__ort_math_backend_recip_float32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(recip, float32)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(neon, recip, float64) {
    const float64* va = (const float64*)a;
    float64* res      = (float64*)result;
    const size_t mw = 2; // 2 float64 per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_recip_float64_fallback;
    }

    // Vectorized loop - process 2 float64 at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float64x2_t ma = vld1q_f64(&va[i]);
        float64x2_t one = vdupq_n_f64(1.0);
        float64x2_t mr = vdivq_f64(one, ma);  // 1.0 / x
        vst1q_f64(&res[i], mr);
    }

__ort_math_backend_recip_float64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(recip, float64)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}