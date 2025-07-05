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
    const float* da = (const float*)a;
    float* dr = (float*)result;

    size_t simd_count = ort_math_simd_optimal_count(count, 8);  /* 8 floats per AVX2 register */

    if (simd_count == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_sqrt_float_fallback;
    }

    /* Vectorized loop - process 8 floats at once */
    for (size_t i = 0; i < simd_count; i += 8) {
        __m256 va = _mm256_loadu_ps(&da[i]);
        __m256 vr = _mm256_sqrt_ps(va);
        _mm256_storeu_ps(&dr[i], vr);
    }

__ort_math_simd_sqrt_float_fallback:
    /* Handle remaining elements with scalar operations */
    if (simd_count < count) {
        ort_math_ops_sqrt_float(
            dr + simd_count,
            da + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_sqrt_double(void* result, const void* a, size_t count) {
    const double* da = (const double*)a;
    double* dr = (double*)result;
    
    size_t simd_count = ort_math_simd_optimal_count(count, 4);  /* 4 doubles per AVX2 register */
    
    if (simd_count == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_sqrt_double_fallback;
    }

    /* Vectorized loop - process 4 doubles at once */
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256d va = _mm256_loadu_pd(&da[i]);
        __m256d vr = _mm256_sqrt_pd(va);
        _mm256_storeu_pd(&dr[i], vr);
    }

__ort_math_simd_sqrt_double_fallback:
    /* Handle remaining elements with scalar operations */
    if (simd_count < count) {
        ort_math_ops_sqrt_double(
            dr + simd_count,
            da + simd_count,
            count - simd_count);
    }
}
