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

#include <immintrin.h>  /* AVX512 */

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, int8_t) {
    const int8_t* va = (const int8_t*)a;
    const int8_t* vb = (const int8_t*)b;
    int8_t* res = (int8_t*)result;
    const size_t mw = 64; // 64 int8_t per AVX512 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_int8_fallback;
    }

    /* Vectorized loop - process 64 int8_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m512i ma = _mm512_load_si512((const __m512i*)&va[i]);
        __m512i mb = _mm512_load_si512((const __m512i*)&vb[i]);
        __m512i mr = _mm512_sub_epi8(ma, mb);
        _mm512_store_si512((__m512i*)&res[i], mr);
    }

__ort_math_backend_sub_int8_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, int8_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, int16_t) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;
    const size_t mw = 32; // 32 int16_t per AVX512 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_int16_fallback;
    }

    /* Vectorized loop - process 32 int16_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m512i ma = _mm512_load_si512((const __m512i*)&va[i]);
        __m512i mb = _mm512_load_si512((const __m512i*)&vb[i]);
        __m512i mr = _mm512_sub_epi16(ma, mb);
        _mm512_store_si512((__m512i*)&res[i], mr);
    }

__ort_math_backend_sub_int16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, int16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, int32_t) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;
    const size_t mw = 16; // 16 int32_t per AVX512 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_int32_fallback;
    }

    /* Vectorized loop - process 16 int32_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m512i ma = _mm512_load_si512((const __m512i*)&va[i]);
        __m512i mb = _mm512_load_si512((const __m512i*)&vb[i]);
        __m512i mr = _mm512_sub_epi32(ma, mb);
        _mm512_store_si512((__m512i*)&res[i], mr);
    }

__ort_math_backend_sub_int32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, int32_t)(
            res    + mc,
            va     + mc,
            vb     + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, int64_t) {
    const int64_t* va = (const int64_t*)a;
    const int64_t* vb = (const int64_t*)b;
    int64_t* res = (int64_t*)result;
    const size_t mw = 8; // 8 int64_t per AVX512 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_int64_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m512i ma = _mm512_load_si512((const __m512i*)&va[i]);
        __m512i mb = _mm512_load_si512((const __m512i*)&vb[i]);
        __m512i mr = _mm512_sub_epi64(ma, mb);
        _mm512_store_si512((__m512i*)&res[i], mr);
    }

__ort_math_backend_sub_int64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, int64_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, uint8_t) {
    const uint8_t* va = (const uint8_t*)a;
    const uint8_t* vb = (const uint8_t*)b;
    uint8_t* res = (uint8_t*)result;
    const size_t mw = 64; // 64 uint8_t per AVX512 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_uint8_fallback;
    }

    /* Vectorized loop - process 64 uint8_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m512i ma = _mm512_load_si512((const __m512i*)&va[i]);
        __m512i mb = _mm512_load_si512((const __m512i*)&vb[i]);
        __m512i mr = _mm512_sub_epi8(ma, mb);
        _mm512_store_si512((__m512i*)&res[i], mr);
    }

__ort_math_backend_sub_uint8_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, uint8_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, uint16_t) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;
    const size_t mw = 32; // 32 uint16_t per AVX512 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_uint16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m512i ma = _mm512_load_si512((const __m512i*)&va[i]);
        __m512i mb = _mm512_load_si512((const __m512i*)&vb[i]);
        __m512i mr = _mm512_sub_epi16(ma, mb);
        _mm512_store_si512((__m512i*)&res[i], mr);
    }

__ort_math_backend_sub_uint16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, uint16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, uint32_t) {
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;
    const size_t mw = 16; // 16 uint32_t per AVX512 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_uint32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m512i ma = _mm512_load_si512((const __m512i*)&va[i]);
        __m512i mb = _mm512_load_si512((const __m512i*)&vb[i]);
        __m512i mr = _mm512_sub_epi32(ma, mb);
        _mm512_store_si512((__m512i*)&res[i], mr);
    }

__ort_math_backend_sub_uint32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, uint32_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, float) {
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    float* res = (float*)result;
    const size_t mw = 16; // 16 floats per AVX512 register

    size_t mc = ort_math_backend_optimal_count(count, mw);  /* 16 floats per AVX512 register */

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sub_float_fallback;
    }

    /* Vectorized loop - process 16 floats at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m512 ma = _mm512_load_ps(&va[i]);
        __m512 mb = _mm512_load_ps(&vb[i]);
        __m512 mr = _mm512_sub_ps(ma, mb);
        _mm512_store_ps(&res[i], mr);
    }

__ort_math_backend_sub_float_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, float)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, double)  {
    const double* va = (const double*)a;
    const double* vb = (const double*)b;
    double* res = (double*)result;
    const size_t mw = 8; // 8 doubles per AVX512 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_double_fallback;
    }

    /* Vectorized loop - process 8 doubles at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m512d ma = _mm512_load_pd(&va[i]);
        __m512d mb = _mm512_load_pd(&vb[i]);
        __m512d mr = _mm512_sub_pd(ma, mb);
        _mm512_store_pd(&res[i], mr);
    }

__ort_math_backend_sub_double_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, double)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}