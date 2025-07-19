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

#include <emmintrin.h> /* SSE2 */

ORT_MATH_BACKEND_MATMUL_OP_DECL(float) {
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
                float tmp[4];
                _mm_storeu_ps(tmp, mr);
                for (int s = 0; s < 4; ++s) sum += tmp[s];
            }
        }
        for (; k < a_cols; k++) {
            sum += va[k] * vb[k * b_cols + j];
        }
        res[j] = sum;
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(double) {
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
                double tmp[2];
                _mm_storeu_pd(tmp, mr);
                for (int s = 0; s < 2; ++s) sum += tmp[s];
            }
        }
        for (; k < a_cols; k++) {
            sum += va[k] * vb[k * b_cols + j];
        }
        res[j] = sum;
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(int16_t) {
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
                __m128i prod = _mm_madd_epi16(ma, mb);
                int32_t tmp[4];
                _mm_storeu_si128((__m128i *)tmp, prod);
                for (int s = 0; s < 4; ++s) sum += tmp[s];
            }
        }
        for (; k < a_cols; k++) {
            sum += (int32_t)va[k] * (int32_t)vb[k * b_cols + j];
        }
        if (sum > INT16_MAX) {
            sum = INT16_MAX;
        } else if (sum < INT16_MIN) {
            sum = INT16_MIN;
        }
        res[j] = (int16_t)sum;
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(uint16_t) {
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
                __m128i prod = _mm_madd_epi16(ma, mb);
                uint32_t tmp[4];
                _mm_storeu_si128((__m128i *)tmp, prod);
                for (int s = 0; s < 4; ++s) sum += tmp[s];
            }
        }
        for (; k < a_cols; k++) {
            sum += (uint32_t)va[k] * (uint32_t)vb[k * b_cols + j];
        }
        if (sum > UINT16_MAX) {
            sum = UINT16_MAX;
        }
        res[j] = (uint16_t)sum;
    }
}