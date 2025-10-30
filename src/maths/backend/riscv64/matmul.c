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

#include "maths/backend/riscv64/impl.h"
#include "maths/backend/riscv64/util.h"

#include <riscv_vector.h> /* RVV */

ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, float) {
    const float *va = (const float *)a;
    const float *vb = (const float *)b;
    float *res = (float *)result;
    for (size_t j = 0; j < b_cols; j++) {
        vfloat32m1_t ma, mb, mr;
        float sum = 0.0f;
        const size_t mw = __riscv_vsetvlmax_e32m1();
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        if (mc > 0) {
            for (; k < mc; k += mw) {
                ma = __riscv_vle32_v_f32m1(&va[k], mw);
                mb = __riscv_vlse32_v_f32m1(&vb[k * b_cols + j], b_cols * sizeof(float), mw);
                mr = __riscv_vfmul_vv_f32m1(ma, mb, mw);

                sum += ORT_MATH_BACKEND_UTIL(riscv64, hsum, float32xN, float)(mr, mw);
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

ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, double) {
    const double *va = (const double *)a;
    const double *vb = (const double *)b;
    double *res = (double *)result;
    for (size_t j = 0; j < b_cols; j++) {
        vfloat64m1_t ma, mb, mr;
        double sum = 0.0;
        const size_t mw = __riscv_vsetvlmax_e64m1();
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        if (mc > 0) {
            for (; k < mc; k += mw) {
                ma = __riscv_vle64_v_f64m1(&va[k], mw);
                mb = __riscv_vlse64_v_f64m1(&vb[k * b_cols + j], b_cols * sizeof(double), mw);
                mr = __riscv_vfmul_vv_f64m1(ma, mb, mw);

                sum += ORT_MATH_BACKEND_UTIL(riscv64, hsum, float64xN, double)(mr, mw);
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

ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, int16_t) {
    const int16_t *va = (const int16_t *)a;
    const int16_t *vb = (const int16_t *)b;
    int16_t *res = (int16_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        int32_t sum = 0;
        const size_t mw = __riscv_vsetvlmax_e16m1();
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        if (mc > 0) {
            for (; k < mc; k += mw) {
                vint16m1_t ma = __riscv_vle16_v_i16m1(&va[k], mw);
                vint16m1_t mb = __riscv_vlse16_v_i16m1(&vb[k * b_cols + j], b_cols * sizeof(int16_t), mw);
                // Widening multiply to 32-bit
                vint32m2_t prod = __riscv_vwmul_vv_i32m2(ma, mb, mw);
                // Reduction sum
                vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
                vint32m1_t sum_vec = __riscv_vredsum_vs_i32m2_i32m1(prod, zero, mw);
                sum += __riscv_vmv_x_s_i32m1_i32(sum_vec);
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

ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, uint16_t) {
    const uint16_t *va = (const uint16_t *)a;
    const uint16_t *vb = (const uint16_t *)b;
    uint16_t *res = (uint16_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        uint32_t sum = 0;
        const size_t mw = __riscv_vsetvlmax_e16m1();
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        if (mc > 0) {
            for (; k < mc; k += mw) {
                vuint16m1_t ma = __riscv_vle16_v_u16m1(&va[k], mw);
                vuint16m1_t mb = __riscv_vlse16_v_u16m1(&vb[k * b_cols + j], b_cols * sizeof(uint16_t), mw);
                // Widening multiply to 32-bit
                vuint32m2_t prod = __riscv_vwmulu_vv_u32m2(ma, mb, mw);
                // Reduction sum
                vuint32m1_t zero = __riscv_vmv_v_x_u32m1(0, 1);
                vuint32m1_t sum_vec = __riscv_vredsum_vs_u32m2_u32m1(prod, zero, mw);
                sum += __riscv_vmv_x_s_u32m1_u32(sum_vec);
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

ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, int32_t) {
    const int32_t *va = (const int32_t *)a;
    const int32_t *vb = (const int32_t *)b;
    int32_t *res = (int32_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        vint32m1_t ma, mb, mr;
        int32_t sum = 0;
        const size_t mw = __riscv_vsetvlmax_e32m1();
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        if (mc > 0) {
            for (; k < mc; k += mw) {
                ma = __riscv_vle32_v_i32m1(&va[k], mw);
                mb = __riscv_vlse32_v_i32m1(&vb[k * b_cols + j], b_cols * sizeof(int32_t), mw);
                mr = __riscv_vmul_vv_i32m1(ma, mb, mw);

                sum += ORT_MATH_BACKEND_UTIL(riscv64, hsum, int32xN, int32_t)(mr, mw);
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

ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, uint32_t) {
    const uint32_t *va = (const uint32_t *)a;
    const uint32_t *vb = (const uint32_t *)b;
    uint32_t *res = (uint32_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        vuint32m1_t ma, mb, mr;
        uint32_t sum = 0;
        const size_t mw = __riscv_vsetvlmax_e32m1();
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        if (mc > 0) {
            for (; k < mc; k += mw) {
                ma = __riscv_vle32_v_u32m1(&va[k], mw);
                mb = __riscv_vlse32_v_u32m1(&vb[k * b_cols + j], b_cols * sizeof(uint32_t), mw);
                mr = __riscv_vmul_vv_u32m1(ma, mb, mw);

                sum += ORT_MATH_BACKEND_UTIL(riscv64, hsum, uint32xN, uint32_t)(mr, mw);
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