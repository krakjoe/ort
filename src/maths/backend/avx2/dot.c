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

static zend_always_inline float
    ort_math_backend_dot_hsum_float(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(low, high);

    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    return _mm_cvtss_f32(sum128);
}

static zend_always_inline double
    ort_math_backend_dot_hsum_double(__m256d v) {
    __m128d low = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1);
    __m128d sum128 = _mm_add_pd(low, high);

    sum128 = _mm_hadd_pd(sum128, sum128);

    return _mm_cvtsd_f64(sum128);
}

static zend_always_inline int32_t
    ort_math_backend_dot_hsum_int32_t(__m256i v) {
    // Convert to 32-bit to avoid overflow during horizontal sum
    __m128i low = _mm256_castsi256_si128(v);
    __m128i high = _mm256_extracti128_si256(v, 1);
    
    // Unpack to 32-bit integers to prevent overflow
    __m256i low_32 = _mm256_cvtepi16_epi32(low);
    __m256i high_32 = _mm256_cvtepi16_epi32(high);
    
    // Add the two 256-bit vectors
    __m256i sum256 = _mm256_add_epi32(low_32, high_32);
    
    // Horizontal sum of 8 32-bit integers
    __m128i sum_low = _mm256_castsi256_si128(sum256);
    __m128i sum_high = _mm256_extracti128_si256(sum256, 1);
    __m128i sum128 = _mm_add_epi32(sum_low, sum_high);
    
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    
    return _mm_cvtsi128_si32(sum128);
}

static zend_always_inline int64_t
    ort_math_backend_dot_hsum_int64_t(__m256i v) {
    __m128i low = _mm256_castsi256_si128(v);
    __m128i high = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(low, high);
    
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    
    return (int64_t)_mm_cvtsi128_si32(sum128);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(dot, float) {
    const float* va = (const float*) a;
    const float* vb = (const float*) b;
    float* res      = (float*) result;
    const size_t mw = 8;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    float sum = 0.0f;

    if (mc > 0) {
        __m256 vsum = _mm256_setzero_ps();
        for (size_t i = 0; i < mc; i += mw) {
            __m256 ma = _mm256_load_ps(&va[i]);
            __m256 mb = _mm256_load_ps(&vb[i]);
            __m256 mr = _mm256_mul_ps(ma, mb);
            vsum = _mm256_add_ps(vsum, mr);
        }
        sum = ort_math_backend_dot_hsum_float(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += va[i] * vb[i];
    }

    res[0] = sum;
}

ORT_MATH_BACKEND_BINARY_OP_DECL(dot, double) {
    const double* va = (const double*) a;
    const double* vb = (const double*) b;
    double* res      = (double*) result;
    const size_t mw = 4;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    double sum = 0.0;

    if (mc > 0) {
        __m256d vsum = _mm256_setzero_pd();
        for (size_t i = 0; i < mc; i += mw) {
            __m256d ma = _mm256_load_pd(&va[i]);
            __m256d mb = _mm256_load_pd(&vb[i]);
            __m256d mr = _mm256_mul_pd(ma, mb);
            vsum = _mm256_add_pd(vsum, mr);
        }
        sum = ort_math_backend_dot_hsum_double(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += va[i] * vb[i];
    }

    res[0] = sum;
}

ORT_MATH_BACKEND_BINARY_OP_DECL(dot, int16_t) {
    const int16_t* va = (const int16_t*) a;
    const int16_t* vb = (const int16_t*) b;
    int16_t* res      = (int16_t*) result;
    const size_t mw = 16;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    int32_t sum = 0;

    if (mc > 0) {
        __m256i vsum = _mm256_setzero_si256();
        for (size_t i = 0; i < mc; i += mw) {
            __m256i ma = _mm256_loadu_si256(
                (const __m256i*)&va[i]);
            __m256i mb = _mm256_loadu_si256(
                (const __m256i*)&vb[i]);
            __m256i mr = _mm256_mullo_epi16(ma, mb);
            vsum = _mm256_add_epi16(vsum, mr);
        }
        sum = ort_math_backend_dot_hsum_int32_t(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += (int32_t)va[i] * (int32_t)vb[i];
    }

    res[0] = ORT_MATH_CLAMP(sum, INT16, int16_t);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(dot, int32_t) {
    const int32_t* va = (const int32_t*) a;
    const int32_t* vb = (const int32_t*) b;
    int32_t* res      = (int32_t*) result;
    const size_t mw = 8;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    int64_t sum = 0;

    if (mc > 0) {
        __m256i vsum = _mm256_setzero_si256();
        for (size_t i = 0; i < mc; i += mw) {
            __m256i ma = _mm256_loadu_si256(
                (const __m256i*)&va[i]);
            __m256i mb = _mm256_loadu_si256(
                (const __m256i*)&vb[i]);
            __m256i mr = _mm256_mullo_epi32(ma, mb);
            vsum = _mm256_add_epi32(vsum, mr);
        }
        sum = ort_math_backend_dot_hsum_int64_t(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += (int64_t)va[i] * (int64_t)vb[i];
    }

    res[0] = ORT_MATH_CLAMP(sum, INT32, int32_t);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(dot, uint16_t) {
    const uint16_t* va = (const uint16_t*) a;
    const uint16_t* vb = (const uint16_t*) b;
    uint16_t* res      = (uint16_t*) result;
    const size_t mw = 16;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    uint32_t sum = 0;

    if (mc > 0) {
        __m256i vsum = _mm256_setzero_si256();
        for (size_t i = 0; i < mc; i += mw) {
            __m256i ma = _mm256_loadu_si256(
                (const __m256i*)&va[i]);
            __m256i mb = _mm256_loadu_si256(
                (const __m256i*)&vb[i]);
            __m256i mr = _mm256_mullo_epi16(ma, mb);
            vsum = _mm256_add_epi16(vsum, mr);
        }
        sum = (uint32_t)ort_math_backend_dot_hsum_int32_t(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += (uint32_t)va[i] * (uint32_t)vb[i];
    }

    res[0] = ORT_MATH_CLAMP_MAX(sum, UINT16, uint16_t);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(dot, uint32_t) {
    const uint32_t* va = (const uint32_t*) a;
    const uint32_t* vb = (const uint32_t*) b;
    uint32_t* res      = (uint32_t*) result;
    const size_t mw = 8;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    uint64_t sum = 0;

    if (mc > 0) {
        __m256i vsum = _mm256_setzero_si256();
        for (size_t i = 0; i < mc; i += mw) {
            __m256i ma = _mm256_loadu_si256(
                (const __m256i*)&va[i]);
            __m256i mb = _mm256_loadu_si256(
                (const __m256i*)&vb[i]);
            __m256i mr = _mm256_mullo_epi32(ma, mb);
            vsum = _mm256_add_epi32(vsum, mr);
        }
        sum = (uint64_t)ort_math_backend_dot_hsum_int64_t(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += (uint64_t)va[i] * (uint64_t)vb[i];
    }

    res[0] = ORT_MATH_CLAMP_MAX(sum, UINT32, uint32_t);
}