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
    const int8_t* va = (const int8_t*)a;
    const int8_t* vb = (const int8_t*)b;
    int8_t* res = (int8_t*)result;
    const size_t mw = 32; // 32 int8_t per AVX2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_sub_int8_fallback;
    }

    /* Vectorized loop - process 32 int8_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_sub_epi8(ma, mb);
        _mm256_storeu_si256((__m256i*)&res[i], mr);
    }

__ort_math_simd_sub_int8_fallback:
    if (mc < count) {
        ort_math_ops_sub_int8_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_sub_int16_t(void* result, const void* a, const void* b, size_t count) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;
    const size_t mw = 16; // 16 int16_t per AVX2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_sub_int16_fallback;
    }

    /* Vectorized loop - process 16 int16_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_sub_epi16(ma, mb);
        _mm256_storeu_si256((__m256i*)&res[i], mr);
    }

__ort_math_simd_sub_int16_fallback:
    if (mc < count) {
        ort_math_ops_sub_int16_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_sub_int32_t(void* result, const void* a, const void* b, size_t count) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;
    const size_t mw = 8; // 8 int32_t per AVX2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_sub_int32_fallback;
    }

    /* Vectorized loop - process 8 int32_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_sub_epi32(ma, mb);
        _mm256_storeu_si256((__m256i*)&res[i], mr);
    }

__ort_math_simd_sub_int32_fallback:
    if (mc < count) {
        ort_math_ops_sub_int32_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_sub_int64_t(void* result, const void* a, const void* b, size_t count) {
    const int64_t* va = (const int64_t*)a;
    const int64_t* vb = (const int64_t*)b;
    int64_t* res = (int64_t*)result;
    const size_t mw = 4; // 4 int64_t per AVX2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_sub_int64_fallback;
    }

    /* Vectorized loop - process 4 int64_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_sub_epi64(ma, mb);
        _mm256_storeu_si256((__m256i*)&res[i], mr);
    }

__ort_math_simd_sub_int64_fallback:
    if (mc < count) {
        ort_math_ops_sub_int64_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_sub_uint8_t(void* result, const void* a, const void* b, size_t count) {
    const uint8_t* va = (const uint8_t*)a;
    const uint8_t* vb = (const uint8_t*)b;
    uint8_t* res = (uint8_t*)result;
    const size_t mw = 32; // 32 uint8_t per AVX2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_sub_uint8_fallback;
    }

    /* Vectorized loop - process 32 uint8_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_sub_epi8(ma, mb);
        _mm256_storeu_si256((__m256i*)&res[i], mr);
    }

__ort_math_simd_sub_uint8_fallback:
    if (mc < count) {
        ort_math_ops_sub_uint8_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_sub_uint16_t(void* result, const void* a, const void* b, size_t count) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;
    const size_t mw = 16; // 16 uint16_t per AVX2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_sub_uint16_fallback;
    }

    /* Vectorized loop - process 16 uint16_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_sub_epi16(ma, mb);
        _mm256_storeu_si256((__m256i*)&res[i], mr);
    }

__ort_math_simd_sub_uint16_fallback:
    if (mc < count) {
        ort_math_ops_sub_uint16_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_sub_uint32_t(void* result, const void* a, const void* b, size_t count) {
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;
    const size_t mw = 8; // 8 uint32_t per AVX2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_sub_uint32_fallback;
    }

    /* Vectorized loop - process 8 uint32_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_sub_epi32(ma, mb);
        _mm256_storeu_si256((__m256i*)&res[i], mr);
    }

__ort_math_simd_sub_uint32_fallback:
    if (mc < count) {
        ort_math_ops_sub_uint32_t(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_sub_float(void* result, const void* a, const void* b, size_t count) {
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    float* res      = (float*)result;
    const size_t mw = 8; // 8 floats per AVX2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_sub_float_fallback;
    }

    /* Vectorized loop - process 8 floats at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256 ma = _mm256_loadu_ps(&va[i]);
        __m256 mb = _mm256_loadu_ps(&vb[i]);
        __m256 mr = _mm256_sub_ps(ma, mb);
        _mm256_storeu_ps(&res[i], mr);
    }

__ort_math_simd_sub_float_fallback:
    if (mc < count) {
        ort_math_ops_sub_float(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

void ort_math_simd_sub_double(void* result, const void* a, const void* b, size_t count) {
    const double* va = (const double*)a;
    const double* vb = (const double*)b;
    double* res = (double*)result;
    const size_t mw = 4; // 4 doubles per AVX2 register

    size_t mc = ort_math_simd_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_simd_sub_double_fallback;
    }

    /* Vectorized loop - process 4 doubles at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256d ma = _mm256_loadu_pd(&va[i]);
        __m256d mb = _mm256_loadu_pd(&vb[i]);
        __m256d mr = _mm256_sub_pd(ma, mb);
        _mm256_storeu_pd(&res[i], mr);
    }

__ort_math_simd_sub_double_fallback:
    if (mc < count) {
        ort_math_ops_sub_double(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}
