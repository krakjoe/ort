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

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, int8_t) {
    const int8_t* va = (const int8_t*)a;
    const int8_t* vb = (const int8_t*)b;
    int8_t* res = (int8_t*)result;
    const size_t mw = 32; // 32 int8_t per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_int8_fallback;
    }

    /* Vectorized loop - process 32 int8_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_load_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_load_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_add_epi8(ma, mb);
        _mm256_store_si256((__m256i*)&res[i], mr);
    }

__ort_math_backend_add_int8_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, int8_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, int16_t) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;
    const size_t mw = 16; // 16 int16_t per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_int16_fallback;
    }

    /* Vectorized loop - process 16 int16_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_load_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_load_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_add_epi16(ma, mb);
        _mm256_store_si256((__m256i*)&res[i], mr);
    }

__ort_math_backend_add_int16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, int16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, int32_t) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;
    const size_t mw = 8; // 8 int32_t per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_int32_fallback;
    }

    /* Vectorized loop - process 8 int32_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_load_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_load_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_add_epi32(ma, mb);
        _mm256_store_si256((__m256i*)&res[i], mr);
    }

__ort_math_backend_add_int32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, int32_t)(
            res    + mc,
            va     + mc,
            vb     + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, int64_t) {
    const int64_t* va = (const int64_t*)a;
    const int64_t* vb = (const int64_t*)b;
    int64_t* res = (int64_t*)result;
    const size_t mw = 4; // 4 int64_t per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_int64_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_load_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_load_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_add_epi64(ma, mb);
        _mm256_store_si256((__m256i*)&res[i], mr);
    }

__ort_math_backend_add_int64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, int64_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, uint8_t) {
    const uint8_t* va = (const uint8_t*)a;
    const uint8_t* vb = (const uint8_t*)b;
    uint8_t* res = (uint8_t*)result;
    const size_t mw = 32; // 32 uint8_t per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_uint8_fallback;
    }

    /* Vectorized loop - process 32 uint8_t at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_load_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_load_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_add_epi8(ma, mb);
        _mm256_store_si256((__m256i*)&res[i], mr);
    }

__ort_math_backend_add_uint8_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, uint8_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, uint16_t) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;
    const size_t mw = 16; // 16 uint16_t per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_uint16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_load_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_load_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_add_epi16(ma, mb);
        _mm256_store_si256((__m256i*)&res[i], mr);
    }

__ort_math_backend_add_uint16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, uint16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, uint32_t) {
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;
    const size_t mw = 8; // 8 uint32_t per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_uint32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m256i ma = _mm256_load_si256((const __m256i*)&va[i]);
        __m256i mb = _mm256_load_si256((const __m256i*)&vb[i]);
        __m256i mr = _mm256_add_epi32(ma, mb);
        _mm256_store_si256((__m256i*)&res[i], mr);
    }

__ort_math_backend_add_uint32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, uint32_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, float16) {
    const float16* va = (const float16*) a;
    const float16* vb = (const float16*) b;
    float16* res      = (float16*)       result;
    const size_t mw = 8; /* AVX2 can process 8 float32 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_add_float16_fallback;
    }

#ifndef ORT_BACKEND_CPU_F16C
    goto __ort_math_backend_add_float16_fallback;
#else
    /* Vectorized loop - process 8 float16 at once */
    for (size_t i = 0; i < mc; i += mw) {
        /* Load 8 float16 values into 128-bit registers */
        __m128i ma = _mm_load_si128((__m128i*)&va[i]);
        __m128i mb = _mm_load_si128((__m128i*)&vb[i]);

        /* Convert F16 to F32: 8 float16 -> 8 float32 */
        __m256 mf16ca = _mm256_cvtph_ps(ma);
        __m256 mf16cb = _mm256_cvtph_ps(mb);

        /* Apply add operation */
        __m256 mf16r = _mm256_add_ps(mf16ca, mf16cb);

        /* Convert F32 back to F16: 8 float32 -> 8 float16 */
        __m128i mr = _mm256_cvtps_ph(
            mf16r, _MM_FROUND_TO_NEAREST_INT);

        /* Store result */
        _mm_store_si128((__m128i*)&res[i], mr);
    }
#endif

__ort_math_backend_add_float16_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, float16)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, float32) {
    const float32* va = (const float32*)a;
    const float32* vb = (const float32*)b;
    float32* res = (float32*)result;
    const size_t mw = 8; // 8 float32 per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);  /* 8 float32 per AVX2 register */

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_add_float32_fallback;
    }

    /* Vectorized loop - process 8 float32 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256 ma = _mm256_load_ps(&va[i]);
        __m256 mb = _mm256_load_ps(&vb[i]);
        __m256 mr = _mm256_add_ps(ma, mb);
        _mm256_store_ps(&res[i], mr);
    }

__ort_math_backend_add_float32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, float32)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, float64)  {
    const float64* va = (const float64*)a;
    const float64* vb = (const float64*)b;
    float64* res = (float64*)result;
    const size_t mw = 4; // 4 float64 per AVX2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_float64_fallback;
    }

    /* Vectorized loop - process 4 float64 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256d ma = _mm256_load_pd(&va[i]);
        __m256d mb = _mm256_load_pd(&vb[i]);
        __m256d mr = _mm256_add_pd(ma, mb);
        _mm256_store_pd(&res[i], mr);
    }

__ort_math_backend_add_float64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, float64)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}