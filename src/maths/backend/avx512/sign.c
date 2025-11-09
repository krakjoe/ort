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

#ifdef ORT_BACKEND_CPU_F16C
ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, sign, float16) {
    const float16* va = (const float16*) a;
    float16* res      = (float16*)       result;
    const size_t mw = 16; /* AVX512 can process 16 float32 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sign_float16_fallback;
    }

    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 neg_one = _mm512_set1_ps(-1.0f);

    /* Vectorized loop - process 16 float16 at once */
    for (size_t i = 0; i < mc; i += mw) {
        /* Load 16 float16 values into 256-bit register */
        __m256i ma = _mm256_load_si256((__m256i*)&va[i]);

        /* Convert F16 to F32: 16 float16 -> 16 float32 */
        __m512 mf16c = _mm512_cvtph_ps(ma);

        /* Check for positive, negative, and zero */
        __mmask16 pos_mask = _mm512_cmp_ps_mask(mf16c, zero, _CMP_GT_OQ);
        __mmask16 neg_mask = _mm512_cmp_ps_mask(mf16c, zero, _CMP_LT_OQ);

        /* Apply sign: positive -> 1, negative -> -1, zero -> 0 */
        __m512 mf16r = _mm512_mask_blend_ps(pos_mask, zero, one);
        mf16r = _mm512_mask_blend_ps(neg_mask, mf16r, neg_one);

        /* Convert F32 back to F16: 16 float32 -> 16 float16 */
        __m256i mr = _mm512_cvtps_ph(
            mf16r, _MM_FROUND_TO_NEAREST_INT);

        /* Store result */
        _mm256_store_si256((__m256i*)&res[i], mr);
    }

    if (mc < count) {
__ort_math_backend_sign_float16_fallback:
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, float16)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
#endif

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, sign, float32) {
    const float32* va = (const float32*)a;
    float32* res = (float32*)result;
    const size_t mw = 16; /* AVX512 can process 16 float32 at once */

    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 neg_one = _mm512_set1_ps(-1.0f);

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sign_float32_fallback;
    }

    /* Vectorized loop - process 16 float32 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m512 ma = _mm512_load_ps(&va[i]);

        /* Check for positive, negative, and zero */
        __mmask16 pos_mask = _mm512_cmp_ps_mask(ma, zero, _CMP_GT_OQ);
        __mmask16 neg_mask = _mm512_cmp_ps_mask(ma, zero, _CMP_LT_OQ);

        /* Apply sign: positive -> 1, negative -> -1, zero -> 0 */
        __m512 mr = _mm512_mask_blend_ps(pos_mask, zero, one);
        mr = _mm512_mask_blend_ps(neg_mask, mr, neg_one);

        _mm512_store_ps(&res[i], mr);
    }

    if (mc < count) {
__ort_math_backend_sign_float32_fallback:
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, float32)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, sign, float64) {
    const float64* va = (const float64*)a;
    float64* res = (float64*)result;
    const size_t mw = 8; /* AVX512 can process 8 float64 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sign_float64_fallback;
    }

    const __m512d zero = _mm512_setzero_pd();
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d neg_one = _mm512_set1_pd(-1.0);

    for (size_t i = 0; i < mc; i += mw) {
        __m512d ma = _mm512_load_pd(&va[i]);

        /* Check for positive, negative, and zero */
        __mmask8 pos_mask = _mm512_cmp_pd_mask(ma, zero, _CMP_GT_OQ);
        __mmask8 neg_mask = _mm512_cmp_pd_mask(ma, zero, _CMP_LT_OQ);

        /* Apply sign: positive -> 1, negative -> -1, zero -> 0 */
        __m512d mr = _mm512_mask_blend_pd(pos_mask, zero, one);
        mr = _mm512_mask_blend_pd(neg_mask, mr, neg_one);

        _mm512_store_pd(&res[i], mr);
    }

    if (mc < count) {
__ort_math_backend_sign_float64_fallback:
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, float64)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
