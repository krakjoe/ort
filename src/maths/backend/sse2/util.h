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

#include <emmintrin.h> /* SSE2 */

ORT_MATH_BACKEND_UTIL_DECL(sse2, 
    hsum, float32x4, float, __m128) {
    float tmp[4];
    
    _mm_storeu_ps(tmp, v);

    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

ORT_MATH_BACKEND_UTIL_DECL(sse2, 
    hsum, float64x2, double, __m128d) {
    return _mm_cvtsd_f64(
        _mm_add_sd(v, _mm_unpackhi_pd(v, v)));
}

ORT_MATH_BACKEND_UTIL_DECL(sse2, 
    hsum, int32x4, int32_t, __m128i) {
    int32_t tmp[4];
    _mm_storeu_si128(
        (__m128i*)tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

ORT_MATH_BACKEND_UTIL_DECL(sse2, 
    hsum, int16x8, int32_t, __m128i) {
    int16_t tmp[8];
    _mm_storeu_si128(
        (__m128i*)tmp, v);
    int32_t sum = 0;
    for (int i = 0; i < 8; ++i) {
        sum += tmp[i];
    }
    return sum;
}
