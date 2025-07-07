/*
  +----------------------------------------------------------------------+
  | ort SIMD AVX2 sqrt implementation                                    |
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

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "maths/simd/impl.h"

void ort_math_simd_sqrt_float(void* result, const void* a, size_t count) {
    const float* va = (const float*)a;
    float* res = (float*)result;
    const size_t mw = 8; /* AVX2 can process 8 floats at once */

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_sqrt_float_fallback;
    }

    /* Vectorized loop - process 8 floats at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256 ma = _mm256_loadu_ps(&va[i]);
        __m256 mr = _mm256_sqrt_ps(ma);
        _mm256_storeu_ps(&res[i], mr);
    }

__ort_math_simd_sqrt_float_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ort_math_ops_sqrt_float(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

void ort_math_simd_sqrt_double(void* result, const void* a, size_t count) {
    const double* va = (const double*)a;
    double* res = (double*)result;
    const size_t mw = 4; /* AVX2 can process 4 doubles at once */
    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_sqrt_double_fallback;
    }

    /* Vectorized loop - process 4 doubles at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256d ma = _mm256_loadu_pd(&va[i]);
        __m256d mr = _mm256_sqrt_pd(ma);
        _mm256_storeu_pd(&res[i], mr);
    }

__ort_math_simd_sqrt_double_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ort_math_ops_sqrt_double(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
