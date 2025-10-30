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
#include "maths/backend/sse41/util.h"

#include <smmintrin.h> /* SSE4.1 */

ORT_MATH_BACKEND_MATMUL_OP_DECL(sse41, float) {
    const float *va = (const float *)a;
    const float *vb = (const float *)b;
    float *res = (float *)result;
    for (size_t j = 0; j < b_cols; j++) {
        __m128 ma, mb, mr;
        float sum = 0.0f;
        const size_t mw = 4;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        if (mc > 0) {
            for (; k < mc; k += mw) {
                ma = _mm_loadu_ps(&va[k]);
                mb = _mm_set_ps(
                    vb[(k+3) * b_cols + j],
                    vb[(k+2) * b_cols + j],
                    vb[(k+1) * b_cols + j],
                    vb[k * b_cols + j]);
                mr = _mm_mul_ps(ma, mb);

                sum += ORT_MATH_BACKEND_UTIL(sse41, hsum, float32x4, float)(mr);
            }
        }
        if (mc < a_cols) {
            /* Fallback to frontend for leftovers */
            for (; k < a_cols; k++) {
                sum += va[k] * vb[k * b_cols + j];
            }
        }
        res[j] = sum;
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(sse41, double) {
    const double *va = (const double *)a;
    const double *vb = (const double *)b;
    double *res = (double *)result;
    for (size_t j = 0; j < b_cols; j++) {
        __m128d ma, mb, mr;
        double sum = 0.0;
        const size_t mw = 2;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        if (mc > 0) {
            for (; k < mc; k += mw) {
                ma = _mm_loadu_pd(&va[k]);
                mb = _mm_set_pd(
                    vb[(k+1) * b_cols + j],
                    vb[k * b_cols + j]);
                mr = _mm_mul_pd(ma, mb);

                sum += ORT_MATH_BACKEND_UTIL(sse41, hsum, float64x2, double)(mr);
            }
        }
        if (mc < a_cols) {
            for (; k < a_cols; k++) {
                sum += va[k] * vb[k * b_cols + j];
            }
        }
        res[j] = sum;
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(sse41, int16_t) {
    const int16_t *va = (const int16_t *)a;
    const int16_t *vb = (const int16_t *)b;
    int16_t *res = (int16_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        int32_t sum = 0;
        const size_t mw = 8;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        if (mc > 0) {
            for (; k < mc; k += mw) {
                __m128i ma = _mm_loadu_si128((const __m128i *)&va[k]);
                __m128i mb = _mm_set_epi16(
                    vb[(k+7) * b_cols + j], vb[(k+6) * b_cols + j],
                    vb[(k+5) * b_cols + j], vb[(k+4) * b_cols + j],
                    vb[(k+3) * b_cols + j], vb[(k+2) * b_cols + j],
                    vb[(k+1) * b_cols + j], vb[k * b_cols + j]);
                // Multiply and horizontally add pairs to 32-bit
                __m128i prod = _mm_madd_epi16(ma, mb);

                sum += ORT_MATH_BACKEND_UTIL(sse41, hsum, int16x8, int32_t)(prod);
            }
        }
        if (mc < a_cols) {
            for (; k < a_cols; k++) {
                sum += (int32_t)va[k] * (int32_t)vb[k * b_cols + j];
            }
        }

        res[j] = ORT_MATH_CLAMP(sum, INT16, int16_t);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(sse41, uint16_t) {
    const uint16_t *va = (const uint16_t *)a;
    const uint16_t *vb = (const uint16_t *)b;
    uint16_t *res = (uint16_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        uint32_t sum = 0;
        const size_t mw = 8;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        if (mc > 0) {
            for (; k < mc; k += mw) {
                __m128i ma = _mm_loadu_si128((const __m128i *)&va[k]);
                __m128i mb = _mm_set_epi16(
                    vb[(k+7) * b_cols + j], vb[(k+6) * b_cols + j],
                    vb[(k+5) * b_cols + j], vb[(k+4) * b_cols + j],
                    vb[(k+3) * b_cols + j], vb[(k+2) * b_cols + j],
                    vb[(k+1) * b_cols + j], vb[k * b_cols + j]);
                // Multiply and horizontally add pairs to 32-bit
                __m128i prod = _mm_madd_epi16(ma, mb);

                sum += ORT_MATH_BACKEND_UTIL(sse41, hsum, int16x8, int32_t)(prod);
            }
        }
        if (mc < a_cols) {
            for (; k < a_cols; k++) {
                sum += (uint32_t)va[k] * (uint32_t)vb[k * b_cols + j];
            }
        }

        res[j] = ORT_MATH_CLAMP_MAX(sum, UINT16, uint16_t);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(sse41, int32_t) {
    const int32_t *va = (const int32_t *)a;
    const int32_t *vb = (const int32_t *)b;
    int32_t *res = (int32_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        __m128i ma, mb, mr;
        int32_t sum = 0;
        const size_t mw = 4;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        if (mc > 0) {
            for (; k < mc; k += mw) {
                ma = _mm_loadu_si128((const __m128i *)&va[k]);
                mb = _mm_set_epi32(
                    vb[(k+3) * b_cols + j],
                    vb[(k+2) * b_cols + j],
                    vb[(k+1) * b_cols + j],
                    vb[k * b_cols + j]);
                mr = _mm_mullo_epi32(ma, mb);

                sum += ORT_MATH_BACKEND_UTIL(sse41, hsum, int32x4, int32_t)(mr);
            }
        }
        if (mc < a_cols) {
            for (; k < a_cols; k++) {
                sum += va[k] * vb[k * b_cols + j];
            }
        }
        res[j] = ORT_MATH_CLAMP(sum, INT32, int32_t);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(sse41, uint32_t) {
    const uint32_t *va = (const uint32_t *)a;
    const uint32_t *vb = (const uint32_t *)b;
    uint32_t *res = (uint32_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        __m128i ma, mb, mr;
        uint32_t sum = 0;
        const size_t mw = 4;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        if (mc > 0) {
            for (; k < mc; k += mw) {
                ma = _mm_loadu_si128((const __m128i *)&va[k]);
                mb = _mm_set_epi32(
                    vb[(k+3) * b_cols + j],
                    vb[(k+2) * b_cols + j],
                    vb[(k+1) * b_cols + j],
                    vb[k * b_cols + j]);
                mr = _mm_mullo_epi32(ma, mb);

                sum += ORT_MATH_BACKEND_UTIL(sse41, hsum, int32x4, int32_t)(mr);
            }
        }
        if (mc < a_cols) {
            for (; k < a_cols; k++) {
                sum += va[k] * vb[k * b_cols + j];
            }
        }
        res[j] = ORT_MATH_CLAMP_MAX(sum, UINT32, uint32_t);
    }
}