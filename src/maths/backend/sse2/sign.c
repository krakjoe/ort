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
 * SIMD Sign Operations (SSE2)
 *
 * Only float and double are contracted for sign in SSE2, matching AVX2/SSE4.1.
 */

void ort_math_backend_sign_float(void* result, const void* a, size_t count) {
    const float* va = (const float*)a;
    float* res = (float*)result;
    const size_t mw = 4; /* SSE2 can process 4 floats at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sign_float_fallback;
    }

    const __m128 zero = _mm_setzero_ps();
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 neg_one = _mm_set1_ps(-1.0f);

    /* Vectorized loop - process 4 floats at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m128 ma = _mm_loadu_ps(&va[i]);
        __m128 pos_mask = _mm_cmpgt_ps(ma, zero);
        __m128 neg_mask = _mm_cmplt_ps(ma, zero);
        __m128 mr = _mm_andnot_ps(
            _mm_or_ps(pos_mask, neg_mask), zero);

        mr = _mm_or_ps(
            mr, _mm_and_ps(pos_mask, one));
        mr = _mm_or_ps(
            mr, _mm_and_ps(neg_mask, neg_one));

        _mm_storeu_ps(&res[i], mr);
    }

__ort_math_backend_sign_float_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ort_math_frontend_sign_float(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

void ort_math_backend_sign_double(void* result, const void* a, size_t count) {
    const double* va = (const double*)a;
    double* res = (double*)result;
    const size_t mw = 2; /* SSE2 can process 2 doubles at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sign_double_fallback;
    }

    const __m128d zero = _mm_setzero_pd();
    const __m128d one = _mm_set1_pd(1.0);
    const __m128d neg_one = _mm_set1_pd(-1.0);

    /* Vectorized loop - process 2 doubles at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m128d ma = _mm_loadu_pd(&va[i]);
        __m128d pos_mask = _mm_cmpgt_pd(ma, zero);
        __m128d neg_mask = _mm_cmplt_pd(ma, zero);
        __m128d mr = _mm_andnot_pd(
            _mm_or_pd(pos_mask, neg_mask), zero);
        mr = _mm_or_pd(mr, 
            _mm_and_pd(pos_mask, one));
        mr = _mm_or_pd(mr, 
            _mm_and_pd(neg_mask, neg_one));
        _mm_storeu_pd(&res[i], mr);
    }

__ort_math_backend_sign_double_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ort_math_frontend_sign_double(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
