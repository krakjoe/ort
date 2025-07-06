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

#include "maths/simd/impl.h"
#include <smmintrin.h> /* SSE4.1 */

void ort_math_simd_round_float(void* result, const void* a, size_t count) {
    const float* va = (const float*)a;
    float* res = (float*)result;
    const size_t simd_width = 4;
    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_round_float_fallback;
    }

    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128 vreg = _mm_loadu_ps(&va[i]);
        __m128 vr = _mm_round_ps(vreg,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_ps(&res[i], vr);
    }

__ort_math_simd_round_float_fallback:
    if (simd_count < count) {
        ort_math_ops_round_float(
            res + simd_count,
            va + simd_count,
            count - simd_count);
    }
}

void ort_math_simd_round_double(void* result, const void* a, size_t count) {
    const double* va = (const double*)a;
    double* res = (double*)result;
    const size_t simd_width = 2;
    size_t simd_count = ort_math_simd_optimal_count(count, simd_width);

    if (simd_count == 0) {
        goto __ort_math_simd_round_double_fallback;
    }

    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m128d vreg = _mm_loadu_pd(&va[i]);
        __m128d vr = _mm_round_pd(vreg,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_pd(&res[i], vr);
    }

__ort_math_simd_round_double_fallback:
    if (simd_count < count) {
        ort_math_ops_round_double(
            res + simd_count,
            va + simd_count,
            count - simd_count);
    }
}
