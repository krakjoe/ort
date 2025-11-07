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

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, trunc, float32) {
    const float32* va = (const float32*)a;
    float32* res      = (float32*)result;
    const size_t mw = 16; // 16 float32 per AVX512 register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_trunc_float32_fallback;
    }

    /* Vectorized loop - process 16 float32 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m512 ma = _mm512_load_ps(&va[i]);
        __m512 mr = _mm512_roundscale_ps(ma, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        _mm512_store_ps(&res[i], mr);
    }

__ort_math_backend_trunc_float32_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(trunc, float32)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, trunc, float64)  {
    const float64* va = (const float64*)a;
    float64* res      = (float64*)result;
    const size_t mw = 8; // 8 float64 per AVX512 register

    size_t mc = ort_math_backend_optimal_count(count, mw);  /* 8 float64 per AVX512 register */

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_trunc_float64_fallback;
    }

    /* Vectorized loop - process 8 float64 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m512d ma = _mm512_load_pd(&va[i]);
        __m512d mr = _mm512_roundscale_pd(ma, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
        _mm512_store_pd(&res[i], mr);
    }

__ort_math_backend_trunc_float64_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(trunc, float64)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}