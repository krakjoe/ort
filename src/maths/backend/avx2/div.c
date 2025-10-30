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

/*
 * SIMD Division Operations (AVX2)
 *
 * Note: AVX2 does not support integer division.
 */

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, div, float) {
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    float* res      = (float*)result;
    const size_t mw = 8; /* AVX2 can process 8 floats at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);  /* 8 floats per AVX2 register */

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_div_float_fallback;
    }

    /* Vectorized loop - process 8 floats at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256 ma = _mm256_load_ps(&va[i]);
        __m256 mb = _mm256_load_ps(&vb[i]);
        __m256 mr = _mm256_div_ps(ma, mb);
        _mm256_store_ps(&res[i], mr);
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

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, div, double) {
    const double* va = (const double*)a;
    const double* vb = (const double*)b;
    double* res = (double*)result;
    const size_t mw = 4; // 4 doubles per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_div_double_fallback;
    }

    /* Vectorized loop - process 4 doubles at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256d ma = _mm256_load_pd(&va[i]);
        __m256d mb = _mm256_load_pd(&vb[i]);
        __m256d mr = _mm256_div_pd(ma, mb);
        _mm256_store_pd(&res[i], mr);
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
