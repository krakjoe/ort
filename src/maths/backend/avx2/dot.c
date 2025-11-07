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
#include "maths/backend/avx2/util.h"

#include <immintrin.h>  /* AVX/AVX2 */

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, float16) {
    const float16* va = (const float16*) a;
    const float16* vb = (const float16*) b;
    float16* res      = (float16*)       result;
    const size_t mw = 8; /* AVX2 can process 8 float32 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);
    float32 sum = 0.0f;

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_dot_float16_fallback;
    }

#ifndef ORT_BACKEND_CPU_F16C
    goto __ort_math_backend_dot_float16_fallback;
#else
    __m256 vsum = _mm256_setzero_ps();

    /* Vectorized loop - process 8 float16 at once */
    for (size_t i = 0; i < mc; i += mw) {
        /* Load 8 float16 values into 128-bit registers */
        __m128i ma = _mm_load_si128((__m128i*)&va[i]);
        __m128i mb = _mm_load_si128((__m128i*)&vb[i]);

        /* Convert F16 to F32: 8 float16 -> 8 float32 */
        __m256 mf16ca = _mm256_cvtph_ps(ma);
        __m256 mf16cb = _mm256_cvtph_ps(mb);

        /* Apply multiply operation and accumulate */
        __m256 mf16r = _mm256_mul_ps(mf16ca, mf16cb);
        vsum = _mm256_add_ps(vsum, mf16r);
    }

    sum = ORT_MATH_BACKEND_UTIL(avx2, hsum, float32x8, float32)(vsum);
#endif

__ort_math_backend_dot_float16_fallback:
    /* Handle remaining elements with scalar operations */
    for (size_t i = mc; i < count; ++i) {
        sum += ort_math_float32_from_float16(va[i]) *
               ort_math_float32_from_float16(vb[i]);
    }

    res[0] = ort_math_float16_from_float32(sum);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, float32) {
    const float32* va = (const float32*) a;
    const float32* vb = (const float32*) b;
    float32* res      = (float32*) result;
    const size_t mw = 8;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    float32 sum = 0.0f;

    if (mc > 0) {
        __m256 vsum = _mm256_setzero_ps();
        for (size_t i = 0; i < mc; i += mw) {
            __m256 ma = _mm256_load_ps(&va[i]);
            __m256 mb = _mm256_load_ps(&vb[i]);
            __m256 mr = _mm256_mul_ps(ma, mb);
            vsum = _mm256_add_ps(vsum, mr);
        }
        sum = ORT_MATH_BACKEND_UTIL(avx2, hsum, float32x8, float32)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += va[i] * vb[i];
    }

    res[0] = sum;
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, float64) {
    const float64* va = (const float64*) a;
    const float64* vb = (const float64*) b;
    float64* res      = (float64*) result;
    const size_t mw = 4;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    float64 sum = 0.0;

    if (mc > 0) {
        __m256d vsum = _mm256_setzero_pd();
        for (size_t i = 0; i < mc; i += mw) {
            __m256d ma = _mm256_load_pd(&va[i]);
            __m256d mb = _mm256_load_pd(&vb[i]);
            __m256d mr = _mm256_mul_pd(ma, mb);
            vsum = _mm256_add_pd(vsum, mr);
        }
        sum = ORT_MATH_BACKEND_UTIL(avx2, hsum, float64x4, float64)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += va[i] * vb[i];
    }

    res[0] = sum;
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, int16_t) {
    const int16_t* va = (const int16_t*) a;
    const int16_t* vb = (const int16_t*) b;
    int16_t* res      = (int16_t*) result;
    const size_t mw = 16;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    int32_t sum = 0;

    if (mc > 0) {
        __m256i vsum = _mm256_setzero_si256();
        for (size_t i = 0; i < mc; i += mw) {
            __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
            __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
            __m256i mr = _mm256_mullo_epi16(ma, mb);
            vsum = _mm256_add_epi16(vsum, mr);
        }
        sum = ORT_MATH_BACKEND_UTIL(avx2, hsum, int16x8, int32_t)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += (int32_t)va[i] * (int32_t)vb[i];
    }

    res[0] = ORT_MATH_CLAMP(sum, INT16, int16_t);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, int32_t) {
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
        sum = ORT_MATH_BACKEND_UTIL(avx2, hsum, int32x8, int64_t)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += (int64_t)va[i] * (int64_t)vb[i];
    }

    res[0] = ORT_MATH_CLAMP(sum, INT32, int32_t);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, uint16_t) {
    const uint16_t* va = (const uint16_t*) a;
    const uint16_t* vb = (const uint16_t*) b;
    uint16_t* res      = (uint16_t*) result;
    const size_t mw = 16;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    uint32_t sum = 0;

    if (mc > 0) {
        __m256i vsum = _mm256_setzero_si256();
        for (size_t i = 0; i < mc; i += mw) {
            __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
            __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
            __m256i mr = _mm256_mullo_epi16(ma, mb);
            vsum = _mm256_add_epi16(vsum, mr);
        }
        sum = ORT_MATH_BACKEND_UTIL(avx2, hsum, int16x8, int32_t)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += (uint32_t)va[i] * (uint32_t)vb[i];
    }

    res[0] = ORT_MATH_CLAMP_MAX(sum, UINT16, uint16_t);
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, uint32_t) {
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
        sum = ORT_MATH_BACKEND_UTIL(avx2, hsum, int32x8, int64_t)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += (uint64_t)va[i] * (uint64_t)vb[i];
    }

    res[0] = ORT_MATH_CLAMP_MAX(sum, UINT32, uint32_t);
}