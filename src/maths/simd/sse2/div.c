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


#include "maths/simd/impl.h"
#include <emmintrin.h> /* SSE2 */

/*
 * SIMD Division Operations (SSE2)
 *
 * Note: SSE2 does not support integer division.
 */


void ort_math_simd_div_float(void* result, const void* a, const void* b, size_t count) {
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    float* res = (float*)result;
    const size_t mw = 4; // 4 floats per SSE2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_div_float_fallback;
    }

    /* Vectorized loop - process 4 floats at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m128 ma = _mm_loadu_ps(&va[i]);
        __m128 mb = _mm_loadu_ps(&vb[i]);
        __m128 mr = _mm_div_ps(ma, mb);
        _mm_storeu_ps(&res[i], mr);
    }

__ort_math_simd_div_float_fallback:
    if (mc < count) {
        ort_math_ops_div_float(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_div_double(void* result, const void* a, const void* b, size_t count) {
    const double* va = (const double*)a;
    const double* vb = (const double*)b;
    double* res = (double*)result;
    const size_t mw = 2; // 2 doubles per SSE2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_div_double_fallback;
    }

    /* Vectorized loop - process 2 doubles at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m128d ma = _mm_loadu_pd(&va[i]);
        __m128d mb = _mm_loadu_pd(&vb[i]);
        __m128d mr = _mm_div_pd(ma, mb);
        _mm_storeu_pd(&res[i], mr);
    }

__ort_math_simd_div_double_fallback:
    if (mc < count) {
        ort_math_ops_div_double(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}
