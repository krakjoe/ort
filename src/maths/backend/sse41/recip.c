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

void ort_math_backend_recip_float(void* result, const void* a, size_t count) {
    const float* va = (const float*)a;
    float* res = (float*)result;
    const size_t mw = 4; // 4 floats per SSE register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_recip_float_fallback;
    }

    __m128 one = _mm_set1_ps(1.0f);

    for (size_t i = 0; i < mc; i += mw) {
        __m128 ma = _mm_loadu_ps(&va[i]);
        __m128 mr = _mm_div_ps(one, ma);
        _mm_storeu_ps(&res[i], mr);
    }

__ort_math_backend_recip_float_fallback:
    if (mc < count) {
        ort_math_frontend_recip_float(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

void ort_math_backend_recip_double(void* result, const void* a, size_t count) {
    const double* va = (const double*)a;
    double* res = (double*)result;
    const size_t mw = 2; // 2 doubles per SSE register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_recip_double_fallback;
    }

    __m128d one = _mm_set1_pd(1.0);

    for (size_t i = 0; i < mc; i += mw) {
        __m128d ma = _mm_loadu_pd(&va[i]);
        __m128d mr = _mm_div_pd(one, ma);
        _mm_storeu_pd(&res[i], mr);
    }

__ort_math_backend_recip_double_fallback:
    if (mc < count) {
        ort_math_frontend_recip_double(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
