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

#include "maths/backend/impl.h"

#include <arm_neon.h>  /* NEON */

ORT_MATH_BACKEND_UNARY_OP_DECL(round, float) {
    const float* va = (const float*)a;
    float* res      = (float*)result;
    const size_t mw = 4; // 4 floats per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_round_float_fallback;
    }

    // Vectorized loop - process 4 floats at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float32x4_t ma = vld1q_f32(&va[i]);
        float32x4_t mr = vrndnq_f32(ma);
        vst1q_f32(&res[i], mr);
    }

__ort_math_backend_round_float_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(round, float)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(round, double) {
    const double* va = (const double*)a;
    double* res      = (double*)result;
    const size_t mw = 2; // 2 doubles per NEON register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_round_double_fallback;
    }

    // Vectorized loop - process 2 doubles at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float64x2_t ma = vld1q_f64(&va[i]);
        float64x2_t mr = vrndnq_f64(ma);
        vst1q_f64(&res[i], mr);
    }

__ort_math_backend_round_double_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(round, double)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}