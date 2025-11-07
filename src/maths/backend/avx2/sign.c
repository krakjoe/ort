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

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, sign, float16) {
    const float16* va = (const float16*) a;
    float16* res      = (float16*)       result;
    const size_t mw = 8; /* AVX2 can process 8 float32 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sign_float16_fallback;
    }

#ifndef ORT_BACKEND_CPU_F16C
    goto __ort_math_backend_sign_float16_fallback;
#else
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);

    /* Vectorized loop - process 8 float16 at once */
    for (size_t i = 0; i < mc; i += mw) {
        /* Load 8 float16 values into 128-bit register */
        __m128i ma = _mm_load_si128((__m128i*)&va[i]);

        /* Convert F16 to F32: 8 float16 -> 8 float32 */
        __m256 mf16c = _mm256_cvtph_ps(ma);

        /* Check for positive, negative, and zero */
        __m256 pos_mask = _mm256_cmp_ps(mf16c, zero, _CMP_GT_OQ);
        __m256 neg_mask = _mm256_cmp_ps(mf16c, zero, _CMP_LT_OQ);

        /* Apply sign: positive -> 1, negative -> -1, zero -> 0 */
        __m256 mf16r = _mm256_blendv_ps(zero, one, pos_mask);
        mf16r = _mm256_blendv_ps(mf16r, neg_one, neg_mask);

        /* Convert F32 back to F16: 8 float32 -> 8 float16 */
        __m128i mr = _mm256_cvtps_ph(
            mf16r, _MM_FROUND_TO_NEAREST_INT);

        /* Store result */
        _mm_store_si128((__m128i*)&res[i], mr);
    }
#endif

__ort_math_backend_sign_float16_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, float16)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, sign, float32) {
    const float32* va = (const float32*)a;
    float32* res = (float32*)result;
    const size_t mw = 8; /* AVX2 can process 8 float32 at once */

    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sign_float32_fallback;
    }

    /* Vectorized loop - process 8 float32 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256 ma = _mm256_load_ps(&va[i]);

        /* Check for positive, negative, and zero */
        __m256 pos_mask = _mm256_cmp_ps(ma, zero, _CMP_GT_OQ);
        __m256 neg_mask = _mm256_cmp_ps(ma, zero, _CMP_LT_OQ);

        /* Apply sign: positive -> 1, negative -> -1, zero -> 0 */
        __m256 mr = _mm256_blendv_ps(zero, one, pos_mask);
        mr = _mm256_blendv_ps(mr, neg_one, neg_mask);

        _mm256_store_ps(&res[i], mr);
    }

__ort_math_backend_sign_float32_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, float32)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, sign, float64) {
    const float64* va = (const float64*)a;
    float64* res = (float64*)result;
    const size_t mw = 4; /* AVX2 can process 4 float64 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sign_float64_fallback;
    }

    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d neg_one = _mm256_set1_pd(-1.0);

    for (size_t i = 0; i < mc; i += mw) {
        __m256d ma = _mm256_load_pd(&va[i]);

        /* Check for positive, negative, and zero */
        __m256d pos_mask = _mm256_cmp_pd(ma, zero, _CMP_GT_OQ);
        __m256d neg_mask = _mm256_cmp_pd(ma, zero, _CMP_LT_OQ);

        /* Apply sign: positive -> 1, negative -> -1, zero -> 0 */
        __m256d mr = _mm256_blendv_pd(zero, one, pos_mask);
        mr = _mm256_blendv_pd(mr, neg_one, neg_mask);

        _mm256_store_pd(&res[i], mr);
    }

__ort_math_backend_sign_float64_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, float64)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}