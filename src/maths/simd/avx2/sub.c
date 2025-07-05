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

void ort_math_simd_sub_int8_t(void* result, const void* a, const void* b, size_t count) {
    const int8_t* pa = (const int8_t*)a;
    const int8_t* pb = (const int8_t*)b;
    int8_t* pr = (int8_t*)result;
    const size_t simd_width = 32; // 32 int8_t per AVX2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_sub_int8_fallback;
    }

    /* Vectorized loop - process 32 int8_t at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m256i va = _mm256_loadu_si256((const __m256i*)&pa[i]);
        __m256i vb = _mm256_loadu_si256((const __m256i*)&pb[i]);
        __m256i vr = _mm256_sub_epi8(va, vb);
        _mm256_storeu_si256((__m256i*)&pr[i], vr);
    }

__ort_math_simd_sub_int8_fallback:
    if (simd_count < count) {
        ort_math_ops_sub_int8_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_sub_int16_t(void* result, const void* a, const void* b, size_t count) {
    const int16_t* pa = (const int16_t*)a;
    const int16_t* pb = (const int16_t*)b;
    int16_t* pr = (int16_t*)result;
    const size_t simd_width = 16; // 16 int16_t per AVX2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_sub_int16_fallback;
    }

    /* Vectorized loop - process 16 int16_t at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m256i va = _mm256_loadu_si256((const __m256i*)&pa[i]);
        __m256i vb = _mm256_loadu_si256((const __m256i*)&pb[i]);
        __m256i vr = _mm256_sub_epi16(va, vb);
        _mm256_storeu_si256((__m256i*)&pr[i], vr);
    }

__ort_math_simd_sub_int16_fallback:
    if (simd_count < count) {
        ort_math_ops_sub_int16_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_sub_int32_t(void* result, const void* a, const void* b, size_t count) {
    const int32_t* pa = (const int32_t*)a;
    const int32_t* pb = (const int32_t*)b;
    int32_t* pr = (int32_t*)result;
    const size_t simd_width = 8; // 8 int32_t per AVX2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_sub_int32_fallback;
    }

    /* Vectorized loop - process 8 int32_t at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m256i va = _mm256_loadu_si256((const __m256i*)&pa[i]);
        __m256i vb = _mm256_loadu_si256((const __m256i*)&pb[i]);
        __m256i vr = _mm256_sub_epi32(va, vb);
        _mm256_storeu_si256((__m256i*)&pr[i], vr);
    }

__ort_math_simd_sub_int32_fallback:
    if (simd_count < count) {
        ort_math_ops_sub_int32_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_sub_int64_t(void* result, const void* a, const void* b, size_t count) {
    const int64_t* pa = (const int64_t*)a;
    const int64_t* pb = (const int64_t*)b;
    int64_t* pr = (int64_t*)result;
    const size_t simd_width = 4; // 4 int64_t per AVX2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_sub_int64_fallback;
    }

    /* Vectorized loop - process 4 int64_t at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m256i va = _mm256_loadu_si256((const __m256i*)&pa[i]);
        __m256i vb = _mm256_loadu_si256((const __m256i*)&pb[i]);
        __m256i vr = _mm256_sub_epi64(va, vb);
        _mm256_storeu_si256((__m256i*)&pr[i], vr);
    }

__ort_math_simd_sub_int64_fallback:
    if (simd_count < count) {
        ort_math_ops_sub_int64_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_sub_uint8_t(void* result, const void* a, const void* b, size_t count) {
    const uint8_t* pa = (const uint8_t*)a;
    const uint8_t* pb = (const uint8_t*)b;
    uint8_t* pr = (uint8_t*)result;
    const size_t simd_width = 32; // 32 uint8_t per AVX2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_sub_uint8_fallback;
    }

    /* Vectorized loop - process 32 uint8_t at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m256i va = _mm256_loadu_si256((const __m256i*)&pa[i]);
        __m256i vb = _mm256_loadu_si256((const __m256i*)&pb[i]);
        __m256i vr = _mm256_sub_epi8(va, vb);
        _mm256_storeu_si256((__m256i*)&pr[i], vr);
    }

__ort_math_simd_sub_uint8_fallback:
    if (simd_count < count) {
        ort_math_ops_sub_uint8_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_sub_uint16_t(void* result, const void* a, const void* b, size_t count) {
    const uint16_t* pa = (const uint16_t*)a;
    const uint16_t* pb = (const uint16_t*)b;
    uint16_t* pr = (uint16_t*)result;
    const size_t simd_width = 16; // 16 uint16_t per AVX2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_sub_uint16_fallback;
    }

    /* Vectorized loop - process 16 uint16_t at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m256i va = _mm256_loadu_si256((const __m256i*)&pa[i]);
        __m256i vb = _mm256_loadu_si256((const __m256i*)&pb[i]);
        __m256i vr = _mm256_sub_epi16(va, vb);
        _mm256_storeu_si256((__m256i*)&pr[i], vr);
    }

__ort_math_simd_sub_uint16_fallback:
    if (simd_count < count) {
        ort_math_ops_sub_uint16_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_sub_uint32_t(void* result, const void* a, const void* b, size_t count) {
    const uint32_t* pa = (const uint32_t*)a;
    const uint32_t* pb = (const uint32_t*)b;
    uint32_t* pr = (uint32_t*)result;
    const size_t simd_width = 8; // 8 uint32_t per AVX2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_sub_uint32_fallback;
    }

    /* Vectorized loop - process 8 uint32_t at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m256i va = _mm256_loadu_si256((const __m256i*)&pa[i]);
        __m256i vb = _mm256_loadu_si256((const __m256i*)&pb[i]);
        __m256i vr = _mm256_sub_epi32(va, vb);
        _mm256_storeu_si256((__m256i*)&pr[i], vr);
    }

__ort_math_simd_sub_uint32_fallback:
    if (simd_count < count) {
        ort_math_ops_sub_uint32_t(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_sub_float(void* result, const void* a, const void* b, size_t count) {
    const float* fa = (const float*)a;
    const float* fb = (const float*)b;
    float* fr = (float*)result;

    size_t simd_count = ort_math_simd_optimal_count(count, 8);  /* 8 floats per AVX2 register */

    if (simd_count == 0) {
        goto __ort_math_simd_sub_float_fallback;
    }

    /* Vectorized loop - process 8 floats at once */
    for (size_t i = 0; i < simd_count; i += 8) {
        __m256 va = _mm256_loadu_ps(&fa[i]);
        __m256 vb = _mm256_loadu_ps(&fb[i]);
        __m256 vr = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(&fr[i], vr);
    }

__ort_math_simd_sub_float_fallback:
    if (simd_count < count) {
        ort_math_ops_sub_float(
            fr + simd_count,
            fa + simd_count,
            fb + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_sub_double(void* result, const void* a, const void* b, size_t count) {
    const double* pa = (const double*)a;
    const double* pb = (const double*)b;
    double* pr = (double*)result;
    const size_t simd_width = 4; // 4 doubles per AVX2 register

    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_sub_double_fallback;
    }

    /* Vectorized loop - process 4 doubles at once */
    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m256d va = _mm256_loadu_pd(&pa[i]);
        __m256d vb = _mm256_loadu_pd(&pb[i]);
        __m256d vr = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(&pr[i], vr);
    }

__ort_math_simd_sub_double_fallback:
    if (simd_count < count) {
        ort_math_ops_sub_double(
            pr + simd_count,
            pa + simd_count,
            pb + simd_count,
            count - simd_count);
    }
}
