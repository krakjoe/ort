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
    const int8_t* pa = (const int8_t*)a;
    const int8_t* pb = (const int8_t*)b;
    int8_t* pr = (int8_t*)result;
    const size_t simd_width = 16; // 16 int8_t per SSE2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_simd_add_int8_fallback;
    }

    /* Vectorized loop - process 16 int8_t at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128i va = _mm_loadu_si128((const __m128i*)&pa[i]);
        __m128i vb = _mm_loadu_si128((const __m128i*)&pb[i]);
        __m128i vr = _mm_add_epi8(va, vb);
        _mm_storeu_si128((__m128i*)&pr[i], vr);
    }

__ort_math_simd_add_int8_fallback:
    if (simd_count < count) {
        ort_math_ops_add_int8_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_add_int16_t(void* result, const void* a, const void* b, size_t count) {
    const int16_t* pa = (const int16_t*)a;
    const int16_t* pb = (const int16_t*)b;
    int16_t* pr = (int16_t*)result;
    const size_t simd_width = 8; // 8 int16_t per SSE2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_add_int16_fallback;
    }

    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128i va = _mm_loadu_si128((const __m128i*)&pa[i]);
        __m128i vb = _mm_loadu_si128((const __m128i*)&pb[i]);
        __m128i vr = _mm_add_epi16(va, vb);
        _mm_storeu_si128((__m128i*)&pr[i], vr);
    }

__ort_math_simd_add_int16_fallback:
    if (simd_count < count) {
        ort_math_ops_add_int16_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_add_int32_t(void* result, const void* a, const void* b, size_t count) {
    const int32_t* pa = (const int32_t*)a;
    const int32_t* pb = (const int32_t*)b;
    int32_t* pr = (int32_t*)result;
    const size_t simd_width = 4; // 4 int32_t per SSE2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_add_int32_fallback;
    }

    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128i va = _mm_loadu_si128((const __m128i*)&pa[i]);
        __m128i vb = _mm_loadu_si128((const __m128i*)&pb[i]);
        __m128i vr = _mm_add_epi32(va, vb);
        _mm_storeu_si128((__m128i*)&pr[i], vr);
    }

__ort_math_simd_add_int32_fallback:
    if (simd_count < count) {
        ort_math_ops_add_int32_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_add_int64_t(void* result, const void* a, const void* b, size_t count) {
    /* No native SSE2 64-bit integer add, fallback only */
    ort_math_ops_add_int64_t(result, a, b, count);
}

void ort_math_simd_add_uint8_t(void* result, const void* a, const void* b, size_t count) {
    const uint8_t* pa = (const uint8_t*)a;
    const uint8_t* pb = (const uint8_t*)b;
    uint8_t* pr = (uint8_t*)result;
    const size_t simd_width = 16; // 16 uint8_t per SSE2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_add_uint8_fallback;
    }

    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128i va = _mm_loadu_si128((const __m128i*)&pa[i]);
        __m128i vb = _mm_loadu_si128((const __m128i*)&pb[i]);
        __m128i vr = _mm_add_epi8(va, vb);
        _mm_storeu_si128((__m128i*)&pr[i], vr);
    }

__ort_math_simd_add_uint8_fallback:
    if (simd_count < count) {
        ort_math_ops_add_uint8_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_add_uint16_t(void* result, const void* a, const void* b, size_t count) {
    const uint16_t* pa = (const uint16_t*)a;
    const uint16_t* pb = (const uint16_t*)b;
    uint16_t* pr = (uint16_t*)result;
    const size_t simd_width = 8; // 8 uint16_t per SSE2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_add_uint16_fallback;
    }

    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128i va = _mm_loadu_si128((const __m128i*)&pa[i]);
        __m128i vb = _mm_loadu_si128((const __m128i*)&pb[i]);
        __m128i vr = _mm_add_epi16(va, vb);
        _mm_storeu_si128((__m128i*)&pr[i], vr);
    }

__ort_math_simd_add_uint16_fallback:
    if (simd_count < count) {
        ort_math_ops_add_uint16_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_add_uint32_t(void* result, const void* a, const void* b, size_t count) {
    const uint32_t* pa = (const uint32_t*)a;
    const uint32_t* pb = (const uint32_t*)b;
    uint32_t* pr = (uint32_t*)result;
    const size_t simd_width = 4; // 4 uint32_t per SSE2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_add_uint32_fallback;
    }

    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128i va = _mm_loadu_si128((const __m128i*)&pa[i]);
        __m128i vb = _mm_loadu_si128((const __m128i*)&pb[i]);
        __m128i vr = _mm_add_epi32(va, vb);
        _mm_storeu_si128((__m128i*)&pr[i], vr);
    }

__ort_math_simd_add_uint32_fallback:
    if (simd_count < count) {
        ort_math_ops_add_uint32_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_add_float(void* result, const void* a, const void* b, size_t count) {
    const float* fa = (const float*)a;
    const float* fb = (const float*)b;
    float* fr = (float*)result;
    const size_t simd_width = 4; // 4 floats per SSE2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_add_float_fallback;
    }

    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128 va = _mm_loadu_ps(&fa[i]);
        __m128 vb = _mm_loadu_ps(&fb[i]);
        __m128 vr = _mm_add_ps(va, vb);
        _mm_storeu_ps(&fr[i], vr);
    }

__ort_math_simd_add_float_fallback:
    if (simd_count < count) {
        ort_math_ops_add_float(
            fr + simd_count,
            fa + simd_count,
            fb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_add_double(void* result, const void* a, const void* b, size_t count) {
    const double* pa = (const double*)a;
    const double* pb = (const double*)b;
    double* pr = (double*)result;
    const size_t simd_width = 2; // 2 doubles per SSE2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_add_double_fallback;
    }

    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128d va = _mm_loadu_pd(&pa[i]);
        __m128d vb = _mm_loadu_pd(&pb[i]);
        __m128d vr = _mm_add_pd(va, vb);
        _mm_storeu_pd(&pr[i], vr);
    }

__ort_math_simd_add_double_fallback:
    if (simd_count < count) {
        ort_math_ops_add_double(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}
