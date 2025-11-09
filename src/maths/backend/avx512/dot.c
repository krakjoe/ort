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
#include "maths/backend/avx512/util.h"

#include <immintrin.h>  /* AVX512 */

#ifdef ORT_BACKEND_CPU_F16C
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, float16) {
    const float16* va = (const float16*) a;
    const float16* vb = (const float16*) b;
    float16* res      = (float16*)       result;
    const size_t mw = 16; /* AVX512 can process 16 float32 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);
    float32 sum = 0.0f;

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_dot_float16_fallback;
    }

    __m512 vsum = _mm512_setzero_ps();

    /* Vectorized loop - process 16 float16 at once */
    for (size_t i = 0; i < mc; i += mw) {
        /* Load 16 float16 values into 256-bit registers */
        __m256i ma = _mm256_load_si256((__m256i*)&va[i]);
        __m256i mb = _mm256_load_si256((__m256i*)&vb[i]);

        /* Convert F16 to F32: 16 float16 -> 16 float32 */
        __m512 mf16ca = _mm512_cvtph_ps(ma);
        __m512 mf16cb = _mm512_cvtph_ps(mb);

        /* Apply multiply operation and accumulate */
        __m512 mf16r = _mm512_mul_ps(mf16ca, mf16cb);
        vsum = _mm512_add_ps(vsum, mf16r);
    }

    sum = ORT_MATH_BACKEND_UTIL(avx512, hsum, float32x16, float32)(vsum);

__ort_math_backend_dot_float16_fallback:
    for (size_t i = mc; i < count; ++i) {
        sum += ort_math_float32_from_float16(va[i]) *
               ort_math_float32_from_float16(vb[i]);
    }

    res[0] = ort_math_float16_from_float32(sum);
}
#endif

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, float32) {
    const float32* va = (const float32*) a;
    const float32* vb = (const float32*) b;
    float32* res      = (float32*) result;
    const size_t mw = 16;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    float32 sum = 0.0f;

    if (mc > 0) {
        __m512 vsum = _mm512_setzero_ps();
        for (size_t i = 0; i < mc; i += mw) {
            __m512 ma = _mm512_load_ps(&va[i]);
            __m512 mb = _mm512_load_ps(&vb[i]);
            __m512 mr = _mm512_mul_ps(ma, mb);
            vsum = _mm512_add_ps(vsum, mr);
        }
        sum = ORT_MATH_BACKEND_UTIL(avx512, hsum, float32x16, float32)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += va[i] * vb[i];
    }

    res[0] = sum;
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, float64) {
    const float64* va = (const float64*) a;
    const float64* vb = (const float64*) b;
    float64* res      = (float64*) result;
    const size_t mw = 8;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    float64 sum = 0.0;

    if (mc > 0) {
        __m512d vsum = _mm512_setzero_pd();
        for (size_t i = 0; i < mc; i += mw) {
            __m512d ma = _mm512_load_pd(&va[i]);
            __m512d mb = _mm512_load_pd(&vb[i]);
            __m512d mr = _mm512_mul_pd(ma, mb);
            vsum = _mm512_add_pd(vsum, mr);
        }
        sum = ORT_MATH_BACKEND_UTIL(avx512, hsum, float64x8, float64)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += va[i] * vb[i];
    }

    res[0] = sum;
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, int16_t) {
    const int16_t* va = (const int16_t*) a;
    const int16_t* vb = (const int16_t*) b;
    int16_t* res      = (int16_t*) result;
    const size_t mw = 32;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    int32_t sum = 0;

    if (mc > 0) {
        __m512i vsum = _mm512_setzero_si512();
        for (size_t i = 0; i < mc; i += mw) {
            __m512i ma = _mm512_loadu_si512((const __m512i*)&va[i]);
            __m512i mb = _mm512_loadu_si512((const __m512i*)&vb[i]);
            __m512i mr = _mm512_mullo_epi16(ma, mb);
            vsum = _mm512_add_epi16(vsum, mr);
        }
        sum = ORT_MATH_BACKEND_UTIL(avx512, hsum, int16x16, int32_t)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += (int32_t)va[i] * (int32_t)vb[i];
    }

    res[0] = (int16_t)sum;
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, int32_t) {
    const int32_t* va = (const int32_t*) a;
    const int32_t* vb = (const int32_t*) b;
    int32_t* res      = (int32_t*) result;
    const size_t mw = 16;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    int32_t sum = 0;

    if (mc > 0) {
        __m512i vsum = _mm512_setzero_si512();
        for (size_t i = 0; i < mc; i += mw) {
            __m512i ma = _mm512_load_si512((const __m512i*)&va[i]);
            __m512i mb = _mm512_load_si512((const __m512i*)&vb[i]);
            __m512i mr = _mm512_mullo_epi32(ma, mb);
            vsum = _mm512_add_epi32(vsum, mr);
        }
        sum = ORT_MATH_BACKEND_UTIL(avx512, hsum, int32x16, int32_t)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += va[i] * vb[i];
    }

    res[0] = sum;
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, uint16_t) {
    const uint16_t* va = (const uint16_t*) a;
    const uint16_t* vb = (const uint16_t*) b;
    uint16_t* res      = (uint16_t*) result;
    const size_t mw = 32;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    uint32_t sum = 0;

    if (mc > 0) {
        __m512i vsum = _mm512_setzero_si512();
        for (size_t i = 0; i < mc; i += mw) {
            __m512i ma = _mm512_loadu_si512((const __m512i*)&va[i]);
            __m512i mb = _mm512_loadu_si512((const __m512i*)&vb[i]);
            __m512i mr = _mm512_mullo_epi16(ma, mb);
            vsum = _mm512_add_epi16(vsum, mr);
        }
        sum = (uint32_t)ORT_MATH_BACKEND_UTIL(avx512, hsum, int16x16, int32_t)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += (uint32_t)va[i] * (uint32_t)vb[i];
    }

    res[0] = (uint16_t)sum;
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, uint32_t) {
    const uint32_t* va = (const uint32_t*) a;
    const uint32_t* vb = (const uint32_t*) b;
    uint32_t* res      = (uint32_t*) result;
    const size_t mw = 16;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    uint32_t sum = 0;

    if (mc > 0) {
        __m512i vsum = _mm512_setzero_si512();
        for (size_t i = 0; i < mc; i += mw) {
            __m512i ma = _mm512_load_si512((const __m512i*)&va[i]);
            __m512i mb = _mm512_load_si512((const __m512i*)&vb[i]);
            __m512i mr = _mm512_mullo_epi32(ma, mb);
            vsum = _mm512_add_epi32(vsum, mr);
        }
        sum = (uint32_t)ORT_MATH_BACKEND_UTIL(avx512, hsum, int32x16, int32_t)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += va[i] * vb[i];
    }

    res[0] = sum;
}