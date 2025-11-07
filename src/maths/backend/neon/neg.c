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

ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, int8_t) {
    const int8_t* va = (const int8_t*)a;
    int8_t* res      = (int8_t*)result;
    const size_t mw = 16; // 16 int8_t per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_neg_int8_fallback;
    }

    // Vectorized loop - process 16 int8_t at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        int8x16_t ma = vld1q_s8(&va[i]);
        int8x16_t mr = vnegq_s8(ma);
        vst1q_s8(&res[i], mr);
    }

__ort_math_backend_neg_int8_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(neg, int8_t)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, int16_t) {
    const int16_t* va = (const int16_t*)a;
    int16_t* res      = (int16_t*)result;
    const size_t mw = 8; // 8 int16_t per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_neg_int16_fallback;
    }

    // Vectorized loop - process 8 int16_t at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        int16x8_t ma = vld1q_s16(&va[i]);
        int16x8_t mr = vnegq_s16(ma);
        vst1q_s16(&res[i], mr);
    }

__ort_math_backend_neg_int16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(neg, int16_t)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, int32_t) {
    const int32_t* va = (const int32_t*)a;
    int32_t* res      = (int32_t*)result;
    const size_t mw = 4; // 4 int32_t per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_neg_int32_fallback;
    }

    // Vectorized loop - process 4 int32_t at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        int32x4_t ma = vld1q_s32(&va[i]);
        int32x4_t mr = vnegq_s32(ma);
        vst1q_s32(&res[i], mr);
    }

__ort_math_backend_neg_int32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(neg, int32_t)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, int64_t) {
    const int64_t* va = (const int64_t*)a;
    int64_t* res      = (int64_t*)result;
    const size_t mw = 2; // 2 int64_t per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_neg_int64_fallback;
    }

    // Vectorized loop - process 2 int64_t at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        int64x2_t ma = vld1q_s64(&va[i]);
        int64x2_t mr = vnegq_s64(ma);
        vst1q_s64(&res[i], mr);
    }

__ort_math_backend_neg_int64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(neg, int64_t)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, float16) {
    const float16* va = (const float16*)a;
    float16* res      = (float16*)result;
    const size_t mw = 8; // 8 float16 per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_neg_float16_fallback;
    }

    // Vectorized loop - process 8 float16 at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float16x8_t ma = vld1q_f16(
            (const float16_t*)&va[i]);
        float16x8_t mr = vnegq_f16(ma);
        vst1q_f16((float16_t*)&res[i], mr);
    }

__ort_math_backend_neg_float16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(neg, float16)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
#endif

ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, float32) {
    const float32* va = (const float32*)a;
    float32* res      = (float32*)result;
    const size_t mw = 4; // 4 float32 per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_neg_float32_fallback;
    }

    // Vectorized loop - process 4 float32 at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float32x4_t ma = vld1q_f32(&va[i]);
        float32x4_t mr = vnegq_f32(ma);
        vst1q_f32(&res[i], mr);
    }

__ort_math_backend_neg_float32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(neg, float32)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, float64) {
    const float64* va = (const float64*)a;
    float64* res      = (float64*)result;
    const size_t mw = 2; // 2 float64 per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_neg_float64_fallback;
    }

    // Vectorized loop - process 2 float64 at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float64x2_t ma = vld1q_f64(&va[i]);
        float64x2_t mr = vnegq_f64(ma);
        vst1q_f64(&res[i], mr);
    }

__ort_math_backend_neg_float64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(neg, float64)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}