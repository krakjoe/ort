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

#include <immintrin.h>  /* AVX/AVX2 */

void ort_math_simd_ceil_float(void* result, const void* a, size_t count) {
    const float* src = (const float*)a;
    float* dst = (float*)result;
    
    size_t simd_count = ort_math_simd_optimal_count(count, 8);  /* 8 floats per AVX2 register */
    
    if (simd_count == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_ceil_float_fallback;
    }

    /* Vectorized loop - process 8 floats at once */
    for (size_t i = 0; i < simd_count; i += 8) {
        __m256 vec = _mm256_loadu_ps(&src[i]);
        __m256 result_vec = _mm256_ceil_ps(vec);
        _mm256_storeu_ps(&dst[i], result_vec);
    }

__ort_math_simd_ceil_float_fallback:
    /* Handle remaining elements with scalar operations */
    if (simd_count < count) {
        ort_math_ops_ceil_float(
            dst + simd_count,
            src + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_ceil_double(void* result, const void* a, size_t count) {
    const double* src = (const double*)a;
    double* dst = (double*)result;
    
    size_t simd_count = ort_math_simd_optimal_count(count, 4);  /* 4 doubles per AVX2 register */
    
    if (simd_count == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_ceil_double_fallback;
    }

    /* Vectorized loop - process 4 doubles at once */
    for (size_t i = 0; i < simd_count; i += 4) {
        __m256d vec = _mm256_loadu_pd(&src[i]);
        __m256d result_vec = _mm256_ceil_pd(vec);
        _mm256_storeu_pd(&dst[i], result_vec);
    }

__ort_math_simd_ceil_double_fallback:
    /* Handle remaining elements with scalar operations */
    if (simd_count < count) {
        ort_math_ops_ceil_double(
            dst + simd_count,
            src + simd_count,
            count - simd_count);
    }
}