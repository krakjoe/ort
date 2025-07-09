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

#include <smmintrin.h> /* SSE4.1 */

ORT_MATH_BACKEND_UNARY_OP_DECL(abs, float) {
    const float* va = (const float*)a;
    float* res = (float*)result;
    const size_t mw = 4; /* 4 floats per SSE4.1 register */
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_abs_float_fallback;
    }

    __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));

    for (size_t i = 0; i < mc; i += mw) {
        __m128 ma = _mm_loadu_ps(&va[i]);
        __m128 mr = _mm_and_ps(ma, mask);
        _mm_storeu_ps(&res[i], mr);
    }

__ort_math_backend_abs_float_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(abs, float)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(abs, double) {
    const double* va = (const double*)a;
    double* res = (double*)result;
    const size_t mw = 2; /* 2 doubles per SSE4.1 register */
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_abs_double_fallback;
    }

    __m128d mask = _mm_castsi128_pd(
        _mm_set1_epi64x(0x7fffffffffffffffLL));

    for (size_t i = 0; i < mc; i += mw) {
        __m128d ma = _mm_loadu_pd(&va[i]);
        __m128d mr = _mm_and_pd(ma, mask);
        _mm_storeu_pd(&res[i], mr);
    }

__ort_math_backend_abs_double_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(abs, double)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
