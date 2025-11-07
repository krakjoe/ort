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

#include "maths/backend/neon/impl.h"
#include "maths/backend/neon/util.h"

#include <arm_neon.h>  /* NEON */

ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, float16) {
    const float16 *va = (const float16 *)a;
    const float16 *vb = (const float16 *)b;
    float16 *res = (float16 *)result;
    for (size_t j = 0; j < b_cols; j++) {
        float16x8_t ma, mb, mr;
        float16 sum = 0.0f;
        const size_t mw = 8;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            ma = vld1q_f16(&va[k]);
            mb = (float16x8_t){
                vb[k * b_cols + j],
                vb[(k+1) * b_cols + j],
                vb[(k+2) * b_cols + j],
                vb[(k+3) * b_cols + j],
                vb[(k+4) * b_cols + j],
                vb[(k+5) * b_cols + j],
                vb[(k+6) * b_cols + j],
                vb[(k+7) * b_cols + j]
            };
            mr = vmulq_f16(ma, mb);
            sum += ORT_MATH_BACKEND_UTIL(neon, hsum, float16x8, float16)(mr);
        }
        /* Fallback for leftovers */
        for (; k < a_cols; k++) {
            sum += va[k] * vb[k * b_cols + j];
        }
        res[j] = sum;
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, float32) {
    const float32 *va = (const float32 *)a;
    const float32 *vb = (const float32 *)b;
    float32 *res = (float32 *)result;
    for (size_t j = 0; j < b_cols; j++) {
        float32x4_t ma, mb, mr;
        float32 sum = 0.0f;
        const size_t mw = 4;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            ma = vld1q_f32(&va[k]);
            mb = (float32x4_t){
                vb[k * b_cols + j],
                vb[(k+1) * b_cols + j],
                vb[(k+2) * b_cols + j],
                vb[(k+3) * b_cols + j]
            };
            mr = vmulq_f32(ma, mb);
            sum += ORT_MATH_BACKEND_UTIL(neon, hsum, float32x4, float32)(mr);
        }
        /* Fallback for leftovers */
        for (; k < a_cols; k++) {
            sum += va[k] * vb[k * b_cols + j];
        }
        res[j] = sum;
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, float64) {
    const float64 *va = (const float64 *)a;
    const float64 *vb = (const float64 *)b;
    float64 *res = (float64 *)result;
    for (size_t j = 0; j < b_cols; j++) {
        float64x2_t ma, mb, mr;
        float64 sum = 0.0;
        const size_t mw = 2;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            ma = vld1q_f64(&va[k]);
            mb = (float64x2_t){
                vb[k * b_cols + j],
                vb[(k+1) * b_cols + j]
            };
            mr = vmulq_f64(ma, mb);
            sum += ORT_MATH_BACKEND_UTIL(neon, hsum, float64x2, float64)(mr);
        }
        for (; k < a_cols; k++) {
            sum += va[k] * vb[k * b_cols + j];
        }
        res[j] = sum;
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, int32_t) {
    const int32_t *va = (const int32_t *)a;
    const int32_t *vb = (const int32_t *)b;
    int32_t *res = (int32_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        int32x4_t ma, mb, mr;
        int64_t sum = 0;
        const size_t mw = 4;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            ma = vld1q_s32(&va[k]);
            mb = (int32x4_t){
                vb[k * b_cols + j],
                vb[(k+1) * b_cols + j],
                vb[(k+2) * b_cols + j],
                vb[(k+3) * b_cols + j]
            };
            mr = vmulq_s32(ma, mb);
            sum += ORT_MATH_BACKEND_UTIL(neon, hsum, int32x4, int32_t)(mr);
        }
        for (; k < a_cols; k++) {
            sum += va[k] * vb[k * b_cols + j];
        }
        res[j] = ORT_MATH_CLAMP(sum, INT32, int32_t);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, uint32_t) {
    const uint32_t *va = (const uint32_t *)a;
    const uint32_t *vb = (const uint32_t *)b;
    uint32_t *res = (uint32_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        uint32x4_t ma, mb, mr;
        int64_t sum = 0;
        const size_t mw = 4;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            ma = vld1q_u32(&va[k]);
            mb = (uint32x4_t){
                vb[k * b_cols + j],
                vb[(k+1) * b_cols + j],
                vb[(k+2) * b_cols + j],
                vb[(k+3) * b_cols + j]
            };
            mr = vmulq_u32(ma, mb);
            sum += ORT_MATH_BACKEND_UTIL(neon, hsum, uint32x4, uint32_t)(mr);
        }
        for (; k < a_cols; k++) {
            sum += va[k] * vb[k * b_cols + j];
        }
        res[j] = ORT_MATH_CLAMP_MAX(sum, UINT32, uint32_t);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, int16_t) {
    const int16_t *va = (const int16_t *)a;
    const int16_t *vb = (const int16_t *)b;
    int16_t *res = (int16_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        int32_t sum = 0;
        const size_t mw = 8;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            int16x8_t ma = vld1q_s16(&va[k]);
            int16x8_t mb = (int16x8_t){
                vb[k * b_cols + j], vb[(k+1) * b_cols + j],
                vb[(k+2) * b_cols + j], vb[(k+3) * b_cols + j],
                vb[(k+4) * b_cols + j], vb[(k+5) * b_cols + j],
                vb[(k+6) * b_cols + j], vb[(k+7) * b_cols + j]
            };
            // Use multiply-accumulate with widening to 32-bit
            int32x4_t low = vmull_s16(
                vget_low_s16(ma), vget_low_s16(mb));
            int32x4_t high = vmull_s16(
                vget_high_s16(ma), vget_high_s16(mb));
            int32x4_t total = vaddq_s32(low, high);
            
            sum += ORT_MATH_BACKEND_UTIL(neon, hsum, int32x4, int32_t)(total);
        }
        for (; k < a_cols; k++) {
            sum += (int32_t)va[k] * (int32_t)vb[k * b_cols + j];
        }

        res[j] = ORT_MATH_CLAMP(sum, INT16, int16_t);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, uint16_t) {
    const uint16_t *va = (const uint16_t *)a;
    const uint16_t *vb = (const uint16_t *)b;
    uint16_t *res = (uint16_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        uint32_t sum = 0;
        const size_t mw = 8;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            uint16x8_t ma = vld1q_u16(&va[k]);
            uint16x8_t mb = (uint16x8_t){
                vb[k * b_cols + j], vb[(k+1) * b_cols + j],
                vb[(k+2) * b_cols + j], vb[(k+3) * b_cols + j],
                vb[(k+4) * b_cols + j], vb[(k+5) * b_cols + j],
                vb[(k+6) * b_cols + j], vb[(k+7) * b_cols + j]
            };
            // Use multiply-accumulate with widening to 32-bit
            uint32x4_t low = vmull_u16(
                vget_low_u16(ma), vget_low_u16(mb));
            uint32x4_t high = vmull_u16(
                vget_high_u16(ma), vget_high_u16(mb));
            uint32x4_t total = vaddq_u32(low, high);
            
            sum += ORT_MATH_BACKEND_UTIL(neon, hsum, uint32x4, uint32_t)(total);
        }
        for (; k < a_cols; k++) {
            sum += (uint32_t)va[k] * (uint32_t)vb[k * b_cols + j];
        }
        
        res[j] = ORT_MATH_CLAMP_MAX(sum, UINT16, uint16_t);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, int8_t) {
    const int8_t *va = (const int8_t *)a;
    const int8_t *vb = (const int8_t *)b;
    int8_t *res = (int8_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        int32_t sum = 0;
        const size_t mw = 16;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            int8x16_t ma = vld1q_s8(&va[k]);
            int8x16_t mb = (int8x16_t){
                vb[k * b_cols + j], vb[(k+1) * b_cols + j],
                vb[(k+2) * b_cols + j], vb[(k+3) * b_cols + j],
                vb[(k+4) * b_cols + j], vb[(k+5) * b_cols + j],
                vb[(k+6) * b_cols + j], vb[(k+7) * b_cols + j],
                vb[(k+8) * b_cols + j], vb[(k+9) * b_cols + j],
                vb[(k+10) * b_cols + j], vb[(k+11) * b_cols + j],
                vb[(k+12) * b_cols + j], vb[(k+13) * b_cols + j],
                vb[(k+14) * b_cols + j], vb[(k+15) * b_cols + j]
            };
            
            // Multiply and accumulate with widening
            int16x8_t prod_low = vmull_s8(
                vget_low_s8(ma), vget_low_s8(mb));
            int16x8_t prod_high = vmull_s8(
                vget_high_s8(ma), vget_high_s8(mb));
            
            // Widen to 32-bit and accumulate using pairwise add
            int32x4_t acc_low = vpaddlq_s16(prod_low);
            int32x4_t acc_high = vpaddlq_s16(prod_high);
            int32x4_t total = vaddq_s32(acc_low, acc_high);
            
            sum += ORT_MATH_BACKEND_UTIL(neon, hsum, int32x4, int32_t)(total);
        }
        for (; k < a_cols; k++) {
            sum += (int32_t)va[k] * (int32_t)vb[k * b_cols + j];
        }

        res[j] = ORT_MATH_CLAMP(sum, INT8, int8_t);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, uint8_t) {
    const uint8_t *va = (const uint8_t *)a;
    const uint8_t *vb = (const uint8_t *)b;
    uint8_t *res = (uint8_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        uint32_t sum = 0;
        const size_t mw = 16;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            uint8x16_t ma = vld1q_u8(&va[k]);
            uint8x16_t mb = (uint8x16_t){
                vb[k * b_cols + j], vb[(k+1) * b_cols + j],
                vb[(k+2) * b_cols + j], vb[(k+3) * b_cols + j],
                vb[(k+4) * b_cols + j], vb[(k+5) * b_cols + j],
                vb[(k+6) * b_cols + j], vb[(k+7) * b_cols + j],
                vb[(k+8) * b_cols + j], vb[(k+9) * b_cols + j],
                vb[(k+10) * b_cols + j], vb[(k+11) * b_cols + j],
                vb[(k+12) * b_cols + j], vb[(k+13) * b_cols + j],
                vb[(k+14) * b_cols + j], vb[(k+15) * b_cols + j]
            };
            
            // Multiply and accumulate with widening
            uint16x8_t prod_low = vmull_u8(
                vget_low_u8(ma), vget_low_u8(mb));
            uint16x8_t prod_high = vmull_u8(
                vget_high_u8(ma), vget_high_u8(mb));
            
            // Widen to 32-bit and accumulate using pairwise add
            uint32x4_t acc_low = vpaddlq_u16(prod_low);
            uint32x4_t acc_high = vpaddlq_u16(prod_high);
            uint32x4_t total = vaddq_u32(acc_low, acc_high);
            
            sum += ORT_MATH_BACKEND_UTIL(neon, hsum, uint32x4, uint32_t)(total);
        }
        for (; k < a_cols; k++) {
            sum += (uint32_t)va[k] * (uint32_t)vb[k * b_cols + j];
        }

        res[j] = ORT_MATH_CLAMP_MAX(sum, UINT8, uint8_t);
    }
}