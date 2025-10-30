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

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, div, float) {
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    float* res      = (float*)result;
    const size_t mw = 4; /* 4 floats per NEON register */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_div_float_fallback;
    }

    // Vectorized loop - process 4 floats at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float32x4_t va4 = vld1q_f32(&va[i]);
        float32x4_t vb4 = vld1q_f32(&vb[i]);
        float32x4_t vr4 = vdivq_f32(va4, vb4);
        vst1q_f32(&res[i], vr4);
    }

__ort_math_backend_div_float_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(div, float)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, div, double) {
    const double* va = (const double*)a;
    const double* vb = (const double*)b;
    double* res = (double*)result;
    const size_t mw = 2; // 2 doubles per NEON register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_div_double_fallback;
    }

    // Vectorized loop - process 2 doubles at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float64x2_t ma = vld1q_f64(&va[i]);
        float64x2_t mb = vld1q_f64(&vb[i]);
        float64x2_t mr = vdivq_f64(ma, mb);
        vst1q_f64(&res[i], mr);
    }

__ort_math_backend_div_double_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(div, double)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}