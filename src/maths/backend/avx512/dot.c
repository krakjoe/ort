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

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, float) {
    const float* va = (const float*) a;
    const float* vb = (const float*) b;
    float* res      = (float*) result;
    const size_t mw = 16;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    float sum = 0.0f;

    if (mc > 0) {
        __m512 vsum = _mm512_setzero_ps();
        for (size_t i = 0; i < mc; i += mw) {
            __m512 ma = _mm512_load_ps(&va[i]);
            __m512 mb = _mm512_load_ps(&vb[i]);
            __m512 mr = _mm512_mul_ps(ma, mb);
            vsum = _mm512_add_ps(vsum, mr);
        }
        sum = ORT_MATH_BACKEND_UTIL(avx512, hsum, float32x16, float)(vsum);
    }

    for (size_t i = mc; i < count; ++i) {
        sum += va[i] * vb[i];
    }

    res[0] = sum;
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, double) {
    const double* va = (const double*) a;
    const double* vb = (const double*) b;
    double* res      = (double*) result;
    const size_t mw = 8;

    size_t mc = ort_math_backend_optimal_count(count, mw);
    double sum = 0.0;

    if (mc > 0) {
        __m512d vsum = _mm512_setzero_pd();
        for (size_t i = 0; i < mc; i += mw) {
            __m512d ma = _mm512_load_pd(&va[i]);
            __m512d mb = _mm512_load_pd(&vb[i]);
            __m512d mr = _mm512_mul_pd(ma, mb);
            vsum = _mm512_add_pd(vsum, mr);
        }
        sum = ORT_MATH_BACKEND_UTIL(avx512, hsum, float64x8, double)(vsum);
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