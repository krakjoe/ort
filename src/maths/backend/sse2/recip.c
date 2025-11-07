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

ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, recip, float32) {
    const float32* va = (const float32*)a;
    float32* res = (float32*)result;
    const size_t mw = 4; /* SSE2 can process 4 float32 at once */

    const __m128 one = _mm_set1_ps(1.0f);
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_recip_float32_fallback;
    }

    /* Vectorized loop - process 4 float32 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m128 ma = _mm_load_ps(&va[i]);
        __m128 mr = _mm_div_ps(one, ma);
        _mm_store_ps(&res[i], mr);
    }

__ort_math_backend_recip_float32_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(recip, float32)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, recip, float64) {
    const float64* va = (const float64*)a;
    float64* res = (float64*)result;
    const size_t mw = 2; /* SSE2 can process 2 float64 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_recip_float64_fallback;
    }

    const __m128d one = _mm_set1_pd(1.0);

    /* Vectorized loop - process 2 float64 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m128d ma = _mm_load_pd(&va[i]);
        __m128d mr = _mm_div_pd(one, ma);
        _mm_store_pd(&res[i], mr);
    }

__ort_math_backend_recip_float64_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(recip, float64)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
