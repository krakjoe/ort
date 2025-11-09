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

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, neg, float16) {
    const float16* va = (const float16*) a;
    float16* res      = (float16*)       result;
    const size_t mw = 16; /* AVX512 can process 16 float32 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_neg_float16_fallback;
    }

#ifndef ORT_BACKEND_CPU_F16C
    goto __ort_math_backend_neg_float16_fallback;
#else
    __m512 sign_mask = _mm512_set1_ps(-0.0f);  /* Sign bit mask */

    /* Vectorized loop - process 16 float16 at once */
    for (size_t i = 0; i < mc; i += mw) {
        /* Load 16 float16 values into 256-bit register */
        __m256i ma = _mm256_load_si256((__m256i*)&va[i]);

        /* Convert F16 to F32: 16 float16 -> 16 float32 */
        __m512 mf16c = _mm512_cvtph_ps(ma);

        /* Apply neg operation: flip sign bit */
        __m512 mf16r = _mm512_xor_ps(mf16c, sign_mask);

        /* Convert F32 back to F16: 16 float32 -> 16 float16 */
        __m256i mr = _mm512_cvtps_ph(
            mf16r, _MM_FROUND_TO_NEAREST_INT);

        /* Store result */
        _mm256_store_si256((__m256i*)&res[i], mr);
    }
#endif

__ort_math_backend_neg_float16_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(neg, float16)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, neg, float32) {
    const float32* va = (const float32*)a;
    float32* res = (float32*)result;
    const size_t mw = 16; /* AVX512 can process 16 float32 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_neg_float32_fallback;
    }

    __m512 sign_mask = _mm512_set1_ps(-0.0f);  /* Sign bit mask */

    /* Vectorized loop - process 16 float32 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m512 ma = _mm512_load_ps(&va[i]);
        __m512 mr = _mm512_xor_ps(ma, sign_mask);  /* Flip sign bit */
        _mm512_store_ps(&res[i], mr);
    }

__ort_math_backend_neg_float32_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(neg, float32)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, neg, float64) {
    const float64* va = (const float64*)a;
    float64* res = (float64*)result;
    const size_t mw = 8; /* AVX512 can process 8 float64 at once */
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_neg_float64_fallback;
    }

    __m512d sign_mask = _mm512_set1_pd(-0.0);  /* Sign bit mask */

    /* Vectorized loop - process 8 float64 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m512d ma = _mm512_load_pd(&va[i]);
        __m512d mr = _mm512_xor_pd(ma, sign_mask);  /* Flip sign bit */
        _mm512_store_pd(&res[i], mr);
    }

__ort_math_backend_neg_float64_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(neg, float64)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
