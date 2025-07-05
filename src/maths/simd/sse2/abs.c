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

void ort_math_simd_abs_float(void* result, const void* a, size_t count) {
    const float* src = (const float*)a;
    float* dst = (float*)result;
    const size_t simd_width = 4; /* 4 floats per SSE2 register */
    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_abs_float_fallback;
    }

    const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));

    /* Vectorized loop - process 4 floats at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128 va = _mm_loadu_ps(&src[i]);
        __m128 result_vec = _mm_and_ps(va, abs_mask);
        _mm_storeu_ps(&dst[i], result_vec);
    }

__ort_math_simd_abs_float_fallback:
    /* Handle remaining elements with scalar operations */
    if (simd_count < count) {
        ort_math_ops_abs_float(
            dst + simd_count,
            src + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_abs_double(void* result, const void* a, size_t count) {
    const double* src = (const double*)a;
    double* dst = (double*)result;
    const size_t simd_width = 2; /* 2 doubles per SSE2 register */
    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_abs_double_fallback;
    }

    const __m128d abs_mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));

    /* Vectorized loop - process 2 doubles at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128d va = _mm_loadu_pd(&src[i]);
        __m128d result_vec = _mm_and_pd(va, abs_mask);
        _mm_storeu_pd(&dst[i], result_vec);
    }

__ort_math_simd_abs_double_fallback:
    /* Handle remaining elements with scalar operations */
    if (simd_count < count) {
        ort_math_ops_abs_double(
            dst + simd_count,
            src + simd_count,
            count - simd_count);
    }
}

