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
    const float* fa = (const float*)a;
    const float* fb = (const float*)b;
    float* fr = (float*)result;
    const size_t simd_width = 4; // 4 floats per SSE2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_div_float_fallback;
    }

    /* Vectorized loop - process 4 floats at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128 va = _mm_loadu_ps(&fa[i]);
        __m128 vb = _mm_loadu_ps(&fb[i]);
        __m128 vr = _mm_div_ps(va, vb);
        _mm_storeu_ps(&fr[i], vr);
    }

__ort_math_simd_div_float_fallback:
    if (simd_count < count) {
        ort_math_ops_div_float(
            fr + simd_count,
            fa + simd_count,
            fb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_div_double(void* result, const void* a, const void* b, size_t count) {
    const double* pa = (const double*)a;
    const double* pb = (const double*)b;
    double* pr = (double*)result;
    const size_t simd_width = 2; // 2 doubles per SSE2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_div_double_fallback;
    }

    /* Vectorized loop - process 2 doubles at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128d va = _mm_loadu_pd(&pa[i]);
        __m128d vb = _mm_loadu_pd(&pb[i]);
        __m128d vr = _mm_div_pd(va, vb);
        _mm_storeu_pd(&pr[i], vr);
    }

__ort_math_simd_div_double_fallback:
    if (simd_count < count) {
        ort_math_ops_div_double(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}
