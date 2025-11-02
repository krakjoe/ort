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

#include <immintrin.h>  /* AVX512 */

ORT_MATH_BACKEND_UTIL_DECL(avx512,
    hsum, float32x16, float, __m512) {
    __m256 low = _mm512_castps512_ps256(v);
    __m256 high = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(low, high);

    __m128 sum_low = _mm256_castps256_ps128(sum256);
    __m128 sum_high = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(sum_low, sum_high);

    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    return _mm_cvtss_f32(sum128);
}

ORT_MATH_BACKEND_UTIL_DECL(avx512,
    hsum, float64x8, double, __m512d) {
    __m256d low = _mm512_castpd512_pd256(v);
    __m256d high = _mm512_extractf64x4_pd(v, 1);
    __m256d sum256 = _mm256_add_pd(low, high);

    __m128d sum_low = _mm256_castpd256_pd128(sum256);
    __m128d sum_high = _mm256_extractf128_pd(sum256, 1);
    __m128d sum128 = _mm_add_pd(sum_low, sum_high);

    sum128 = _mm_hadd_pd(sum128, sum128);

    return _mm_cvtsd_f64(sum128);
}

ORT_MATH_BACKEND_UTIL_DECL(avx512,
    hsum, int16x16, int32_t, __m512i) {
    __m256i low = _mm512_castsi512_si256(v);
    __m256i high = _mm512_extracti32x8_epi32(v, 1);
    
    __m128i low_low = _mm256_castsi256_si128(low);
    __m128i low_high = _mm256_extracti128_si256(low, 1);
    __m128i high_low = _mm256_castsi256_si128(high);
    __m128i high_high = _mm256_extracti128_si256(high, 1);
    
    __m256i low_low32 = _mm256_cvtepi16_epi32(low_low);
    __m256i low_high32 = _mm256_cvtepi16_epi32(low_high);
    __m256i high_low32 = _mm256_cvtepi16_epi32(high_low);
    __m256i high_high32 = _mm256_cvtepi16_epi32(high_high);

    __m256i sum_low = _mm256_add_epi32(low_low32, low_high32);
    __m256i sum_high = _mm256_add_epi32(high_low32, high_high32);
    __m256i sum256 = _mm256_add_epi32(sum_low, sum_high);
    
    __m128i sum_low128 = _mm256_castsi256_si128(sum256);
    __m128i sum_high128 = _mm256_extracti128_si256(sum256, 1);
    __m128i sum128 = _mm_add_epi32(sum_low128, sum_high128);
    
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);

    return _mm_cvtsi128_si32(sum128);
}

ORT_MATH_BACKEND_UTIL_DECL(avx512,
    hsum, int32x16, int32_t, __m512i) {
    __m256i low = _mm512_castsi512_si256(v);
    __m256i high = _mm512_extracti32x8_epi32(v, 1);
    __m256i sum256 = _mm256_add_epi32(low, high);
    
    __m128i sum_low = _mm256_castsi256_si128(sum256);
    __m128i sum_high = _mm256_extracti128_si256(sum256, 1);
    __m128i sum128 = _mm_add_epi32(sum_low, sum_high);

    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);

    return _mm_cvtsi128_si32(sum128);
}

ORT_MATH_BACKEND_UTIL_DECL(avx512,
    hsum, int32x16, int64_t, __m512i) {
    __m256i low = _mm512_castsi512_si256(v);
    __m256i high = _mm512_extracti32x8_epi32(v, 1);
    __m256i sum256 = _mm256_add_epi32(low, high);
    
    __m128i sum_low = _mm256_castsi256_si128(sum256);
    __m128i sum_high = _mm256_extracti128_si256(sum256, 1);
    __m128i sum128 = _mm_add_epi32(sum_low, sum_high);

    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);

    return (int64_t)_mm_cvtsi128_si32(sum128);
}