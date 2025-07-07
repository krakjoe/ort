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
 * SIMD Addition Operations (SSE2)
 */


void ort_math_simd_add_int8_t(void* result, const void* a, const void* b, size_t count) {
    const int8_t* va = (const int8_t*)a;
    const int8_t* vb = (const int8_t*)b;
    int8_t* res = (int8_t*)result;
    const size_t mw = 16; // 16 int8_t per SSE2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_add_int8_fallback;
    }

    /* Vectorized loop - process 16 int8_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_loadu_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_loadu_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_add_epi8(ma, mb);
        _mm_storeu_si128((__m128i*)&res[i], mr);
    }

__ort_math_simd_add_int8_fallback:
    if (mc < count) {
        ort_math_ops_add_int8_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_add_int16_t(void* result, const void* a, const void* b, size_t count) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;
    const size_t mw = 8; // 8 int16_t per SSE2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_add_int16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_loadu_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_loadu_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_add_epi16(ma, mb);
        _mm_storeu_si128((__m128i*)&res[i], mr);
    }

__ort_math_simd_add_int16_fallback:
    if (mc < count) {
        ort_math_ops_add_int16_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_add_int32_t(void* result, const void* a, const void* b, size_t count) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;
    const size_t mw = 4; // 4 int32_t per SSE2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_add_int32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_loadu_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_loadu_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_add_epi32(ma, mb);
        _mm_storeu_si128((__m128i*)&res[i], mr);
    }

__ort_math_simd_add_int32_fallback:
    if (mc < count) {
        ort_math_ops_add_int32_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_add_uint8_t(void* result, const void* a, const void* b, size_t count) {
    const uint8_t* va = (const uint8_t*)a;
    const uint8_t* vb = (const uint8_t*)b;
    uint8_t* res = (uint8_t*)result;
    const size_t mw = 16; // 16 uint8_t per SSE2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_add_uint8_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_loadu_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_loadu_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_add_epi8(ma, mb);
        _mm_storeu_si128((__m128i*)&res[i], mr);
    }

__ort_math_simd_add_uint8_fallback:
    if (mc < count) {
        ort_math_ops_add_uint8_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_add_uint16_t(void* result, const void* a, const void* b, size_t count) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;
    const size_t mw = 8; // 8 uint16_t per SSE2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_add_uint16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_loadu_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_loadu_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_add_epi16(ma, mb);
        _mm_storeu_si128((__m128i*)&res[i], mr);
    }

__ort_math_simd_add_uint16_fallback:
    if (mc < count) {
        ort_math_ops_add_uint16_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_add_uint32_t(void* result, const void* a, const void* b, size_t count) {
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;
    const size_t mw = 4; // 4 uint32_t per SSE2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_add_uint32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_loadu_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_loadu_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_add_epi32(ma, mb);
        _mm_storeu_si128((__m128i*)&res[i], mr);
    }

__ort_math_simd_add_uint32_fallback:
    if (mc < count) {
        ort_math_ops_add_uint32_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_add_float(void* result, const void* a, const void* b, size_t count) {
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    float* res = (float*)result;
    const size_t mw = 4; // 4 floats per SSE2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_add_float_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128 ma = _mm_loadu_ps(&va[i]);
        __m128 mb = _mm_loadu_ps(&vb[i]);
        __m128 mr = _mm_add_ps(ma, mb);
        _mm_storeu_ps(&res[i], mr);
    }

__ort_math_simd_add_float_fallback:
    if (mc < count) {
        ort_math_ops_add_float(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_add_double(void* result, const void* a, const void* b, size_t count) {
    const double* va = (const double*)a;
    const double* vb = (const double*)b;
    double* res = (double*)result;
    const size_t mw = 2; // 2 doubles per SSE2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_add_double_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128d ma = _mm_loadu_pd(&va[i]);
        __m128d mb = _mm_loadu_pd(&vb[i]);
        __m128d mr = _mm_add_pd(ma, mb);
        _mm_storeu_pd(&res[i], mr);
    }

__ort_math_simd_add_double_fallback:
    if (mc < count) {
        ort_math_ops_add_double(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}
