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

#include <emmintrin.h> /* SSE2 */

/*
 * SIMD Negation Operations (SSE2)
 *
 * Only float and double are contracted for neg in SSE2, matching AVX2/SSE4.1.
 */

void ort_math_backend_neg_float(void* result, const void* a, size_t count) {
    const float* va = (const float*)a;
    float* res = (float*)result;
    const size_t mw = 4; // 4 floats per SSE2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_neg_float_fallback;
    }

    __m128 neg_mask = _mm_set1_ps(-0.0f);

    for (size_t i = 0; i < mc; i += mw) {
        __m128 ma = _mm_loadu_ps(&va[i]);
        __m128 mr = _mm_xor_ps(ma, neg_mask);
        _mm_storeu_ps(&res[i], mr);
    }

__ort_math_backend_neg_float_fallback:
    if (mc < count) {
        ort_math_frontend_neg_float(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

void ort_math_backend_neg_double(void* result, const void* a, size_t count) {
    const double* va = (const double*)a;
    double* res = (double*)result;
    const size_t mw = 2; // 2 doubles per SSE2 register
    size_t mc = ort_math_backend_optimal_count(count, mw);
    __m128d neg_mask = _mm_set1_pd(-0.0);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_neg_double_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128d ma = _mm_loadu_pd(&va[i]);
        __m128d mr = _mm_xor_pd(ma, neg_mask);
        _mm_storeu_pd(&res[i], mr);
    }

__ort_math_backend_neg_double_fallback:
    if (mc < count) {
        ort_math_frontend_neg_double(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
