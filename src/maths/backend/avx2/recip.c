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

#include <immintrin.h>  /* AVX/AVX2 */

ORT_MATH_BACKEND_UNARY_OP_DECL(recip, float) {
    const float* va = (const float*)a;
    float* res      = (float*)result;
    const size_t mw = 8; /* AVX2 can process 8 floats at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_recip_float_fallback;
    }

    const __m256 one = _mm256_set1_ps(1.0f);

    /* Vectorized loop - process 8 floats at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256 ma = _mm256_loadu_ps(&va[i]);
        __m256 mr = _mm256_div_ps(one, ma);
        _mm256_storeu_ps(&res[i], mr);
    }

__ort_math_backend_recip_float_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(recip, float)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(recip, double) {
    const double* va = (const double*)a;
    double* res      = (double*)result;
    const size_t mw  = 4; /* AVX2 can process 4 doubles at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_recip_double_fallback;
    }

    const __m256d one = _mm256_set1_pd(1.0);

    /* Vectorized loop - process 4 doubles at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256d ma = _mm256_loadu_pd(&va[i]);
        __m256d mr = _mm256_div_pd(one, ma);
        _mm256_storeu_pd(&res[i], mr);
    }

__ort_math_backend_recip_double_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(recip, double)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
