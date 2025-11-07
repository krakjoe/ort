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


#include "maths/backend/sse2/impl.h"
#include <emmintrin.h> /* SSE2 */

/*
 * SIMD Division Operations (SSE2)
 *
 * Note: SSE2 does not support integer division.
 */


ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, div, float32) {
    const float32* va = (const float32*)a;
    const float32* vb = (const float32*)b;
    float32* res = (float32*)result;
    const size_t mw = 4; // 4 float32 per SSE2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_div_float32_fallback;
    }

    /* Vectorized loop - process 4 float32 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m128 ma = _mm_load_ps(&va[i]);
        __m128 mb = _mm_load_ps(&vb[i]);
        __m128 mr = _mm_div_ps(ma, mb);
        _mm_store_ps(&res[i], mr);
    }

__ort_math_backend_div_float32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(div, float32)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, div, float64) {
    const float64* va = (const float64*)a;
    const float64* vb = (const float64*)b;
    float64* res = (float64*)result;
    const size_t mw = 2; // 2 float64 per SSE2 register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_div_float64_fallback;
    }

    /* Vectorized loop - process 2 float64 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m128d ma = _mm_load_pd(&va[i]);
        __m128d mb = _mm_load_pd(&vb[i]);
        __m128d mr = _mm_div_pd(ma, mb);
        _mm_store_pd(&res[i], mr);
    }

__ort_math_backend_div_float64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(div, float64)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}
