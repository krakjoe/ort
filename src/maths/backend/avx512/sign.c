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

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, sign, float) {
    const float* va = (const float*)a;
    float* res = (float*)result;
    const size_t mw = 16; /* AVX512 can process 16 floats at once */

    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 neg_one = _mm512_set1_ps(-1.0f);

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sign_float_fallback;
    }

    /* Vectorized loop - process 16 floats at once */
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

__ort_math_backend_sign_float_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, float)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, sign, double) {
    const double* va = (const double*)a;
    double* res = (double*)result;
    const size_t mw = 8; /* AVX512 can process 8 doubles at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sign_double_fallback;
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

__ort_math_backend_sign_double_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, double)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
