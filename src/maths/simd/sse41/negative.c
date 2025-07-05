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

#include <smmintrin.h>
#include <emmintrin.h>
#include <stdint.h>
#include <stddef.h>
#include "../impl.h"

/*
 * SSE4.1 SIMD implementation for elementwise negation (float/double only)
 */

#define SSE41_NEG_IMPL(type, sse_type, load, store, xor_mask) \
void ort_math_simd_negative_##type(void* result, const void* a, size_t count) { \
    size_t i = 0; \
    size_t simd_width = sizeof(sse_type) / sizeof(type); \
    size_t simd_count = ort_math_simd_optimal_count(count, simd_width); \
    for (; i < simd_count; i += simd_width) { \
        sse_type va = load((const void*)((const type*)a + i)); \
        sse_type vr = _mm_xor_ps(va, xor_mask); \
        store((void*)((type*)result + i), vr); \
    } \
    for (; i < count; ++i) { \
        ((type*)result)[i] = -((const type*)a)[i]; \
    } \
}

static const __m128 neg_mask_ps = { -0.0f, -0.0f, -0.0f, -0.0f };
static const __m128d neg_mask_pd = { -0.0, -0.0 };

SSE41_NEG_IMPL(float, __m128, _mm_loadu_ps, _mm_storeu_ps, neg_mask_ps)
SSE41_NEG_IMPL(double, __m128d, _mm_loadu_pd, _mm_storeu_pd, neg_mask_pd)
