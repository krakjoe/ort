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

ORT_MATH_BACKEND_UTIL_DECL(avx2,
    hsum, float32x8, float, __m256) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(low, high);

    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    return _mm_cvtss_f32(sum128);
}

ORT_MATH_BACKEND_UTIL_DECL(avx2,
    hsum, float64x4, double, __m256d) {
    __m128d low = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1);
    __m128d sum128 = _mm_add_pd(low, high);

    sum128 = _mm_hadd_pd(sum128, sum128);

    return _mm_cvtsd_f64(sum128);
}

ORT_MATH_BACKEND_UTIL_DECL(avx2,
    hsum, int16x8, int32_t, __m256i) {
    __m128i low = _mm256_castsi256_si128(v);
    __m128i high = _mm256_extracti128_si256(v, 1);
    __m256i low32 = _mm256_cvtepi16_epi32(low);
    __m256i high32 = _mm256_cvtepi16_epi32(high);

    __m256i sum256 = _mm256_add_epi32(low32, high32);
    __m128i sum_low = _mm256_castsi256_si128(sum256);
    __m128i sum_high = _mm256_extracti128_si256(sum256, 1);
    __m128i sum128 = _mm_add_epi32(sum_low, sum_high);
    
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);

    return _mm_cvtsi128_si32(sum128);
}

ORT_MATH_BACKEND_UTIL_DECL(avx2,
    hsum, int32x8, int32_t, __m256i) {
    __m128i sum_low = _mm256_castsi256_si128(v);
    __m128i sum_high = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(sum_low, sum_high);

    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);

    return _mm_cvtsi128_si32(sum128);
}

ORT_MATH_BACKEND_UTIL_DECL(avx2,
    hsum, int32x8, int64_t, __m256i) {
    __m128i low = _mm256_castsi256_si128(v);
    __m128i high = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(low, high);

    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);

    return (int64_t)_mm_cvtsi128_si32(sum128);
}