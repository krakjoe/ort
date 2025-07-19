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
#include "maths/backend/avx2/util.h"

#include <immintrin.h> /* AVX2 */

ORT_MATH_BACKEND_MATMUL_OP_DECL(float) {
    const float *va = (const float *)a;
    const float *vb = (const float *)b;
    float *res = (float *)result;
    for (size_t j = 0; j < b_cols; j++) {
        __m256 ma, mb, mr;
        float sum = 0.0f;
        const size_t mw = 8;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            ma = _mm256_loadu_ps(&va[k]);
            mb = _mm256_set_ps(
                vb[(k+7) * b_cols + j], vb[(k+6) * b_cols + j],
                vb[(k+5) * b_cols + j], vb[(k+4) * b_cols + j],
                vb[(k+3) * b_cols + j], vb[(k+2) * b_cols + j],
                vb[(k+1) * b_cols + j], vb[k * b_cols + j]);
            mr = _mm256_mul_ps(ma, mb);
            sum += ort_math_backend_hsum_float(mr);
        }
        /* Fallback for leftovers */
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
        __m256d ma, mb, mr;
        double sum = 0.0;
        const size_t mw = 4;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            ma = _mm256_loadu_pd(&va[k]);
            mb = _mm256_set_pd(
                vb[(k+3) * b_cols + j], vb[(k+2) * b_cols + j],
                vb[(k+1) * b_cols + j], vb[k * b_cols + j]);
            mr = _mm256_mul_pd(ma, mb);
            sum += ort_math_backend_hsum_double(mr);
        }
        for (; k < a_cols; k++) {
            sum += va[k] * vb[k * b_cols + j];
        }
        res[j] = sum;
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(int32_t) {
    const int32_t *va = (const int32_t *)a;
    const int32_t *vb = (const int32_t *)b;
    int32_t *res = (int32_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        __m256i ma, mb, mr;
        int32_t sum = 0;
        const size_t mw = 8;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            ma = _mm256_loadu_si256((const __m256i *)&va[k]);
            mb = _mm256_set_epi32(
                vb[(k+7) * b_cols + j], vb[(k+6) * b_cols + j],
                vb[(k+5) * b_cols + j], vb[(k+4) * b_cols + j],
                vb[(k+3) * b_cols + j], vb[(k+2) * b_cols + j],
                vb[(k+1) * b_cols + j], vb[k * b_cols + j]);
            mr = _mm256_mullo_epi32(ma, mb);
            sum += ort_math_backend_hsum_int32_t(mr);
        }
        for (; k < a_cols; k++) {
            sum += va[k] * vb[k * b_cols + j];
        }
        res[j] = sum;
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(uint32_t) {
    const uint32_t *va = (const uint32_t *)a;
    const uint32_t *vb = (const uint32_t *)b;
    uint32_t *res = (uint32_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        __m256i ma, mb, mr;
        uint32_t sum = 0;
        const size_t mw = 8;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            ma = _mm256_loadu_si256((const __m256i *)&va[k]);
            mb = _mm256_set_epi32(
                vb[(k+7) * b_cols + j], vb[(k+6) * b_cols + j],
                vb[(k+5) * b_cols + j], vb[(k+4) * b_cols + j],
                vb[(k+3) * b_cols + j], vb[(k+2) * b_cols + j],
                vb[(k+1) * b_cols + j], vb[k * b_cols + j]);
            mr = _mm256_mullo_epi32(ma, mb);
            sum += (uint32_t)ort_math_backend_hsum_int32_t((__m256i)mr);
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
        const size_t mw = 16;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            __m256i ma = _mm256_loadu_si256((const __m256i *)&va[k]);
            __m256i mb = _mm256_set_epi16(
                vb[(k+15) * b_cols + j], vb[(k+14) * b_cols + j],
                vb[(k+13) * b_cols + j], vb[(k+12) * b_cols + j],
                vb[(k+11) * b_cols + j], vb[(k+10) * b_cols + j],
                vb[(k+9) * b_cols + j], vb[(k+8) * b_cols + j],
                vb[(k+7) * b_cols + j], vb[(k+6) * b_cols + j],
                vb[(k+5) * b_cols + j], vb[(k+4) * b_cols + j],
                vb[(k+3) * b_cols + j], vb[(k+2) * b_cols + j],
                vb[(k+1) * b_cols + j], vb[k * b_cols + j]);
            // Multiply and horizontally add pairs to 32-bit
            __m256i prod = _mm256_madd_epi16(ma, mb);
            sum += ort_math_backend_hsum_int32_t(prod);
        }
        for (; k < a_cols; k++) {
            sum += (int32_t)va[k] * (int32_t)vb[k * b_cols + j];
        }

        res[j] = ORT_MATH_CLAMP(sum, INT16, int16_t);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(uint16_t) {
    const uint16_t *va = (const uint16_t *)a;
    const uint16_t *vb = (const uint16_t *)b;
    uint16_t *res = (uint16_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        uint32_t sum = 0;
        const size_t mw = 16;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            __m256i ma = _mm256_loadu_si256((const __m256i *)&va[k]);
            __m256i mb = _mm256_set_epi16(
                vb[(k+15) * b_cols + j], vb[(k+14) * b_cols + j],
                vb[(k+13) * b_cols + j], vb[(k+12) * b_cols + j],
                vb[(k+11) * b_cols + j], vb[(k+10) * b_cols + j],
                vb[(k+9) * b_cols + j], vb[(k+8) * b_cols + j],
                vb[(k+7) * b_cols + j], vb[(k+6) * b_cols + j],
                vb[(k+5) * b_cols + j], vb[(k+4) * b_cols + j],
                vb[(k+3) * b_cols + j], vb[(k+2) * b_cols + j],
                vb[(k+1) * b_cols + j], vb[k * b_cols + j]);
            // Multiply and horizontally add pairs to 32-bit
            __m256i prod = _mm256_madd_epi16(ma, mb);
            sum += (uint32_t)ort_math_backend_hsum_int32_t(prod);
        }
        for (; k < a_cols; k++) {
            sum += (uint32_t)va[k] * (uint32_t)vb[k * b_cols + j];
        }

        res[j] = ORT_MATH_CLAMP_MAX(sum, UINT16, uint16_t);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(int8_t) {
    const int8_t *va = (const int8_t *)a;
    const int8_t *vb = (const int8_t *)b;
    int8_t *res = (int8_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        int32_t sum = 0;
        const size_t mw = 32;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            __m256i ma = _mm256_loadu_si256((const __m256i *)&va[k]);
            __m256i mb = _mm256_set_epi8(
                vb[(k+31) * b_cols + j], vb[(k+30) * b_cols + j],
                vb[(k+29) * b_cols + j], vb[(k+28) * b_cols + j],
                vb[(k+27) * b_cols + j], vb[(k+26) * b_cols + j],
                vb[(k+25) * b_cols + j], vb[(k+24) * b_cols + j],
                vb[(k+23) * b_cols + j], vb[(k+22) * b_cols + j],
                vb[(k+21) * b_cols + j], vb[(k+20) * b_cols + j],
                vb[(k+19) * b_cols + j], vb[(k+18) * b_cols + j],
                vb[(k+17) * b_cols + j], vb[(k+16) * b_cols + j],
                vb[(k+15) * b_cols + j], vb[(k+14) * b_cols + j],
                vb[(k+13) * b_cols + j], vb[(k+12) * b_cols + j],
                vb[(k+11) * b_cols + j], vb[(k+10) * b_cols + j],
                vb[(k+9) * b_cols + j], vb[(k+8) * b_cols + j],
                vb[(k+7) * b_cols + j], vb[(k+6) * b_cols + j],
                vb[(k+5) * b_cols + j], vb[(k+4) * b_cols + j],
                vb[(k+3) * b_cols + j], vb[(k+2) * b_cols + j],
                vb[(k+1) * b_cols + j], vb[k * b_cols + j]);
            // Multiply and horizontally add pairs to 16-bit
            __m256i prod = _mm256_maddubs_epi16(ma, mb);
            // Now horizontally add pairs to 32-bit
            __m256i prod32 = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
            sum += ort_math_backend_hsum_8xint32_t(prod32);
        }
        for (; k < a_cols; k++) {
            sum += (int32_t)va[k] * (int32_t)vb[k * b_cols + j];
        }

        res[j] = ORT_MATH_CLAMP(sum, INT8, int8_t);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(uint8_t) {
    const uint8_t *va = (const uint8_t *)a;
    const uint8_t *vb = (const uint8_t *)b;
    uint8_t *res = (uint8_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        uint32_t sum = 0;
        const size_t mw = 32;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            __m256i ma = _mm256_loadu_si256((const __m256i *)&va[k]);
            __m256i mb = _mm256_set_epi8(
                vb[(k+31) * b_cols + j], vb[(k+30) * b_cols + j],
                vb[(k+29) * b_cols + j], vb[(k+28) * b_cols + j],
                vb[(k+27) * b_cols + j], vb[(k+26) * b_cols + j],
                vb[(k+25) * b_cols + j], vb[(k+24) * b_cols + j],
                vb[(k+23) * b_cols + j], vb[(k+22) * b_cols + j],
                vb[(k+21) * b_cols + j], vb[(k+20) * b_cols + j],
                vb[(k+19) * b_cols + j], vb[(k+18) * b_cols + j],
                vb[(k+17) * b_cols + j], vb[(k+16) * b_cols + j],
                vb[(k+15) * b_cols + j], vb[(k+14) * b_cols + j],
                vb[(k+13) * b_cols + j], vb[(k+12) * b_cols + j],
                vb[(k+11) * b_cols + j], vb[(k+10) * b_cols + j],
                vb[(k+9) * b_cols + j], vb[(k+8) * b_cols + j],
                vb[(k+7) * b_cols + j], vb[(k+6) * b_cols + j],
                vb[(k+5) * b_cols + j], vb[(k+4) * b_cols + j],
                vb[(k+3) * b_cols + j], vb[(k+2) * b_cols + j],
                vb[(k+1) * b_cols + j], vb[k * b_cols + j]);
            // Multiply and horizontally add pairs to 16-bit
            __m256i prod = _mm256_maddubs_epi16(ma, mb);
            // Now horizontally add pairs to 32-bit
            __m256i prod32 = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
            sum += (uint32_t)ort_math_backend_hsum_8xint32_t(prod32);
        }
        for (; k < a_cols; k++) {
            sum += (uint32_t)va[k] * (uint32_t)vb[k * b_cols + j];
        }

        res[j] = ORT_MATH_CLAMP_MAX(sum, UINT8, uint8_t);
    }
}