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

ORT_MATH_BACKEND_UNARY_OP_DECL(sse41, floor, float32) {
    const float32* va = (const float32*)a;
    float32* res = (float32*)result;
    const size_t mw = 4;
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_floor_float32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128 ma = _mm_load_ps(&va[i]);
        __m128 mr = _mm_floor_ps(ma);
        _mm_store_ps(&res[i], mr);
    }

__ort_math_backend_floor_float32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(floor, float32)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(sse41, floor, float64) {
    const float64* va = (const float64*)a;
    float64* res = (float64*)result;
    const size_t mw = 2; // 2 float64 per SSE register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_floor_float64_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        __m128d ma = _mm_load_pd(&va[i]);
        __m128d mr = _mm_floor_pd(ma);
        _mm_store_pd(&res[i], mr);
    }

__ort_math_backend_floor_float64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(floor, float64)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
