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

#ifdef ORT_BACKEND_CPU_F16C
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, recip, float16) {
    const float16* va = (const float16*) a;
    float16* res      = (float16*)       result;
    const size_t mw = 8; /* AVX2 can process 8 float32 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_recip_float16_fallback;
    }

    const __m256 one = _mm256_set1_ps(1.0f);

    /* Vectorized loop - process 8 float16 at once */
    for (size_t i = 0; i < mc; i += mw) {
        /* Load 8 float16 values into 128-bit register */
        __m128i ma = _mm_load_si128((__m128i*)&va[i]);

        /* Convert F16 to F32: 8 float16 -> 8 float32 */
        __m256 mf16c = _mm256_cvtph_ps(ma);

        /* Apply recip operation: 1.0 / x */
        __m256 mf16r = _mm256_div_ps(one, mf16c);

        /* Convert F32 back to F16: 8 float32 -> 8 float16 */
        __m128i mr = _mm256_cvtps_ph(
            mf16r, _MM_FROUND_TO_NEAREST_INT);

        /* Store result */
        _mm_store_si128((__m128i*)&res[i], mr);
    }

    if (mc < count) {
__ort_math_backend_recip_float16_fallback:
        ORT_MATH_FRONTEND_OP_SYMBOL(recip, float16)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
#endif

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, recip, float32) {
    const float32* va = (const float32*)a;
    float32* res      = (float32*)result;
    const size_t mw = 8; /* AVX2 can process 8 float32 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_recip_float32_fallback;
    }

    const __m256 one = _mm256_set1_ps(1.0f);

    /* Vectorized loop - process 8 float32 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256 ma = _mm256_load_ps(&va[i]);
        __m256 mr = _mm256_div_ps(one, ma);
        _mm256_store_ps(&res[i], mr);
    }

    if (mc < count) {
__ort_math_backend_recip_float32_fallback:
        ORT_MATH_FRONTEND_OP_SYMBOL(recip, float32)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, recip, float64) {
    const float64* va = (const float64*)a;
    float64* res      = (float64*)result;
    const size_t mw  = 4; /* AVX2 can process 4 float64 at once */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_recip_float64_fallback;
    }

    const __m256d one = _mm256_set1_pd(1.0);

    /* Vectorized loop - process 4 float64 at once */
    for (size_t i = 0; i < mc; i += mw) {
        __m256d ma = _mm256_load_pd(&va[i]);
        __m256d mr = _mm256_div_pd(one, ma);
        _mm256_store_pd(&res[i], mr);
    }

    if (mc < count) {
__ort_math_backend_recip_float64_fallback:
        ORT_MATH_FRONTEND_OP_SYMBOL(recip, float64)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
