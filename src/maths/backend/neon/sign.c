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

ORT_MATH_BACKEND_UNARY_OP_DECL(neon, sign, float) {
    const float* va = (const float*)a;
    float* res      = (float*)result;
    const size_t mw = 4; // 4 floats per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sign_float_fallback;
    }

    // Vectorized loop - process 4 floats at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float32x4_t ma = vld1q_f32(&va[i]);
        float32x4_t zero = vdupq_n_f32(0.0f);
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t neg_one = vdupq_n_f32(-1.0f);
        
        // Create masks for positive and negative values
        uint32x4_t pos_mask = vcgtq_f32(ma, zero);  // ma > 0
        uint32x4_t neg_mask = vcltq_f32(ma, zero);  // ma < 0
        
        // Select values based on masks
        float32x4_t result = vbslq_f32(pos_mask, one, zero);
        float32x4_t mr = vbslq_f32(neg_mask, neg_one, result);
        
        vst1q_f32(&res[i], mr);
    }

__ort_math_backend_sign_float_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, float)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(neon, sign, double) {
    const double* va = (const double*)a;
    double* res      = (double*)result;
    const size_t mw = 2; // 2 doubles per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sign_double_fallback;
    }

    // Vectorized loop - process 2 doubles at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float64x2_t ma = vld1q_f64(&va[i]);
        float64x2_t zero = vdupq_n_f64(0.0);
        float64x2_t one = vdupq_n_f64(1.0);
        float64x2_t neg_one = vdupq_n_f64(-1.0);
        
        // Create masks for positive and negative values
        uint64x2_t pos_mask = vcgtq_f64(ma, zero);  // ma > 0
        uint64x2_t neg_mask = vcltq_f64(ma, zero);  // ma < 0
        
        // Select values based on masks
        float64x2_t result = vbslq_f64(pos_mask, one, zero);
        float64x2_t mr = vbslq_f64(neg_mask, neg_one, result);
        
        vst1q_f64(&res[i], mr);
    }

__ort_math_backend_sign_double_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, double)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}