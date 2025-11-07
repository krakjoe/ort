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

#include "maths/backend/sse41/impl.h"

#include <smmintrin.h> /* SSE4.1 */

ORT_MATH_BACKEND_BINARY_OP_DECL(sse41, sub, int8_t) {
    const int8_t* va = (const int8_t*)a;
    const int8_t* vb = (const int8_t*)b;
    int8_t* res = (int8_t*)result;
    const size_t mw = 16; // 16 int8_t per SSE register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_int8_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_load_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_load_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_sub_epi8(ma, mb);
        _mm_store_si128((__m128i*)&res[i], mr);
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

ORT_MATH_BACKEND_BINARY_OP_DECL(sse41, sub, int16_t) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;
    const size_t mw = 8; // 8 int16_t per SSE register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_int16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_load_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_load_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_sub_epi16(ma, mb);
        _mm_store_si128((__m128i*)&res[i], mr);
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

ORT_MATH_BACKEND_BINARY_OP_DECL(sse41, sub, int32_t) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;
    const size_t mw = 4; // 4 int32_t per SSE register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_int32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_load_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_load_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_sub_epi32(ma, mb);
        _mm_store_si128((__m128i*)&res[i], mr);
    }

__ort_math_backend_sub_int32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, int32_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(sse41, sub, int64_t) {
    const int64_t* va = (const int64_t*)a;
    const int64_t* vb = (const int64_t*)b;
    int64_t* res = (int64_t*)result;
    const size_t mw = 2; // 2 int64_t per SSE register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_int64_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_load_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_load_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_sub_epi64(ma, mb);
        _mm_store_si128((__m128i*)&res[i], mr);
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

ORT_MATH_BACKEND_BINARY_OP_DECL(sse41, sub, uint8_t) {
    const uint8_t* va = (const uint8_t*)a;
    const uint8_t* vb = (const uint8_t*)b;
    uint8_t* res = (uint8_t*)result;
    const size_t mw = 16; // 16 uint8_t per SSE register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_uint8_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_load_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_load_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_sub_epi8(ma, mb);
        _mm_store_si128((__m128i*)&res[i], mr);
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

ORT_MATH_BACKEND_BINARY_OP_DECL(sse41, sub, uint16_t) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;
    const size_t mw = 8; // 8 uint16_t per SSE register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_uint16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_load_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_load_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_sub_epi16(ma, mb);
        _mm_store_si128((__m128i*)&res[i], mr);
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

ORT_MATH_BACKEND_BINARY_OP_DECL(sse41, sub, uint32_t) {
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;
    const size_t mw = 4; // 4 uint32_t per SSE register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_uint32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128i ma = _mm_load_si128((const __m128i*)&va[i]);
        __m128i mb = _mm_load_si128((const __m128i*)&vb[i]);
        __m128i mr = _mm_sub_epi32(ma, mb);
        _mm_store_si128((__m128i*)&res[i], mr);
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

ORT_MATH_BACKEND_BINARY_OP_DECL(sse41, sub, float32) {
    const float32* va = (const float32*)a;
    const float32* vb = (const float32*)b;
    float32* res = (float32*)result;
    const size_t mw = 4; // 4 float32 per SSE register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_float32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128 ma = _mm_load_ps(&va[i]);
        __m128 mb = _mm_load_ps(&vb[i]);
        __m128 mr = _mm_sub_ps(ma, mb);
        _mm_store_ps(&res[i], mr);
    }

__ort_math_backend_sub_float32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, float32)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(sse41, sub, float64) {
    const float64* va = (const float64*)a;
    const float64* vb = (const float64*)b;
    float64* res = (float64*)result;
    const size_t mw = 2; // 2 float64 per SSE register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_float64_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128d ma = _mm_load_pd(&va[i]);
        __m128d mb = _mm_load_pd(&vb[i]);
        __m128d mr = _mm_sub_pd(ma, mb);
        _mm_store_pd(&res[i], mr);
    }

__ort_math_backend_sub_float64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, float64)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}
