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

void ort_math_simd_sign_float(void* result, const void* a, size_t count) {
    const float* src = (const float*)a;
    float* dst = (float*)result;
    const size_t simd_width = 8; /* AVX2 can process 8 floats at once */
    
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    
    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_sign_float_fallback;
    }

    /* Vectorized loop - process 8 floats at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m256 va = _mm256_loadu_ps(&src[i]);
        
        /* Check for positive, negative, and zero */
        __m256 pos_mask = _mm256_cmp_ps(va, zero, _CMP_GT_OQ);
        __m256 neg_mask = _mm256_cmp_ps(va, zero, _CMP_LT_OQ);
        
        /* Apply sign: positive -> 1, negative -> -1, zero -> 0 */
        __m256 result_vec = _mm256_blendv_ps(zero, one, pos_mask);
        result_vec = _mm256_blendv_ps(result_vec, neg_one, neg_mask);
        
        _mm256_storeu_ps(&dst[i], result_vec);
    }

__ort_math_simd_sign_float_fallback:
    /* Handle remaining elements with scalar operations */
    if (simd_count < count) {
        ort_math_ops_sign_float(
            dst + simd_count,
            src + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_sign_double(void* result, const void* a, size_t count) {
    const double* src = (const double*)a;
    double* dst = (double*)result;
    const size_t simd_width = 4; /* AVX2 can process 4 doubles at once */
    
    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_sign_double_fallback;
    }

    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d neg_one = _mm256_set1_pd(-1.0);
    
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m256d va = _mm256_loadu_pd(&src[i]);
        
        /* Check for positive, negative, and zero */
        __m256d pos_mask = _mm256_cmp_pd(va, zero, _CMP_GT_OQ);
        __m256d neg_mask = _mm256_cmp_pd(va, zero, _CMP_LT_OQ);
        
        /* Apply sign: positive -> 1, negative -> -1, zero -> 0 */
        __m256d result_vec = _mm256_blendv_pd(zero, one, pos_mask);
        result_vec = _mm256_blendv_pd(result_vec, neg_one, neg_mask);
        
        _mm256_storeu_pd(&dst[i], result_vec);
    }

__ort_math_simd_sign_double_fallback:
    /* Handle remaining elements with scalar operations */
    if (simd_count < count) {
        ort_math_ops_sign_double(
            dst + simd_count,
            src + simd_count,
            count - simd_count);
    }
}