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

ORT_MATH_BACKEND_UNARY_OP_DECL(sse41, sign, float) {
    const float* va = (const float*)a;
    float* res = (float*)result;
    const size_t mw = 4;
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sign_float_fallback;
    }

    __m128 zero = _mm_setzero_ps();
    __m128 one = _mm_set1_ps(1.0f);
    __m128 mone = _mm_set1_ps(-1.0f);

    for (size_t i = 0; i < mc; i += mw) {
        __m128 ma = _mm_load_ps(&va[i]);
        __m128 gt = _mm_cmpgt_ps(ma, zero);
        __m128 lt = _mm_cmplt_ps(ma, zero);
        __m128 vr = _mm_or_ps(
            _mm_and_ps(gt, one), 
            _mm_and_ps(lt, mone));

        _mm_store_ps(&res[i], vr);
    }

__ort_math_backend_sign_float_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, float)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(sse41, sign, double) {
    const double* va = (const double*)a;
    double* res = (double*)result;
    const size_t mw = 2;
    size_t mc = ort_math_backend_optimal_count(count, mw);


    if (mc == 0) {
        goto __ort_math_backend_sign_double_fallback;
    }

    __m128d zero = _mm_setzero_pd();
    __m128d one = _mm_set1_pd(1.0);
    __m128d mone = _mm_set1_pd(-1.0);

    for (size_t i = 0; i < mc; i += mw) {
        __m128d ma = _mm_load_pd(&va[i]);
        __m128d gt = _mm_cmpgt_pd(ma, zero);
        __m128d lt = _mm_cmplt_pd(ma, zero);
        __m128d mr = _mm_or_pd(
            _mm_and_pd(gt, one),
            _mm_and_pd(lt, mone));
        _mm_store_pd(&res[i], mr);
    }

__ort_math_backend_sign_double_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, double)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
