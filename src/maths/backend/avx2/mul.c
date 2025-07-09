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

ORT_MATH_BACKEND_BINARY_OP_DECL(mul, int16_t) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;
    const size_t mw = 16; // 16 int16_t per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_mul_int16_fallback;
    }

    /* Vectorized loop - process 16 int16_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_mullo_epi16(ma, mb);
        _mm256_storeu_si256((__m256i*)&res[i], mr);
    }

__ort_math_backend_mul_int16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(mul, int16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(mul, int32_t) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;
    const size_t mw = 8; // 8 int32_t per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_mul_int32_fallback;
    }

    /* Vectorized loop - process 8 int32_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_mullo_epi32(ma, mb);
        _mm256_storeu_si256((__m256i*)&res[i], mr);
    }

__ort_math_backend_mul_int32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(mul, int32_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(mul, uint16_t) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;
    const size_t mw = 16; // 16 uint16_t per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_mul_uint16_fallback;
    }

    /* Vectorized loop - process 16 uint16_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_mullo_epi16(ma, mb);
        _mm256_storeu_si256((__m256i*)&res[i], mr);
    }

__ort_math_backend_mul_uint16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(mul, uint16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(mul, uint32_t){
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;
    const size_t mw = 8; // 8 uint32_t per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_mul_uint32_fallback;
    }

    /* Vectorized loop - process 8 uint32_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_loadu_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_loadu_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_mullo_epi32(ma, mb);
        _mm256_storeu_si256((__m256i*)&res[i], mr);
    }

__ort_math_backend_mul_uint32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(mul, uint32_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(mul, float) {
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    float* res = (float*)result;
    const size_t mw = 8; // 8 floats per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_mul_float_fallback;
    }

    /* Vectorized loop - process 8 floats at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256 ma = _mm256_loadu_ps(&va[i]);
        __m256 mb = _mm256_loadu_ps(&vb[i]);
        __m256 mr = _mm256_mul_ps(ma, mb);
        _mm256_storeu_ps(&res[i], mr);
    }

__ort_math_backend_mul_float_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(mul, float)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(mul, double) {
    const double* va = (const double*)a;
    const double* vb = (const double*)b;
    double* res = (double*)result;
    const size_t mw = 4; // 4 doubles per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_mul_double_fallback;
    }

    /* Vectorized loop - process 4 doubles at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256d ma = _mm256_loadu_pd(&va[i]);
        __m256d mb = _mm256_loadu_pd(&vb[i]);
        __m256d mr = _mm256_mul_pd(ma, mb);
        _mm256_storeu_pd(&res[i], mr);
    }

__ort_math_backend_mul_double_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(mul, double)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}
