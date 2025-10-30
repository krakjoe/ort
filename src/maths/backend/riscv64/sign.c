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
#include <riscv_vector.h> /* RVV */

ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, sign, float) {
    const float* va = (const float*)a;
    float* res = (float*)result;
    const size_t mw = __riscv_vsetvlmax_e32m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sign_float_fallback;
    }

    const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, mw);
    const vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, mw);
    const vfloat32m1_t neg_one = __riscv_vfmv_v_f_f32m1(-1.0f, mw);

    /* Vectorized loop */
    for (size_t i = 0; i < mc; i += mw) {
        vfloat32m1_t ma = __riscv_vle32_v_f32m1(&va[i], mw);
        vbool32_t pos_mask = __riscv_vmfgt_vf_f32m1_b32(ma, 0.0f, mw);
        vbool32_t neg_mask = __riscv_vmflt_vf_f32m1_b32(ma, 0.0f, mw);
        
        vfloat32m1_t mr = zero;
        mr = __riscv_vmerge_vvm_f32m1(mr, one, pos_mask, mw);
        mr = __riscv_vmerge_vvm_f32m1(mr, neg_one, neg_mask, mw);

        __riscv_vse32_v_f32m1(&res[i], mr, mw);
    }

__ort_math_backend_sign_float_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, float)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, sign, double) {
    const double* va = (const double*)a;
    double* res = (double*)result;
    const size_t mw = __riscv_vsetvlmax_e64m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_sign_double_fallback;
    }

    const vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, mw);
    const vfloat64m1_t one = __riscv_vfmv_v_f_f64m1(1.0, mw);
    const vfloat64m1_t neg_one = __riscv_vfmv_v_f_f64m1(-1.0, mw);

    /* Vectorized loop */
    for (size_t i = 0; i < mc; i += mw) {
        vfloat64m1_t ma = __riscv_vle64_v_f64m1(&va[i], mw);
        vbool64_t pos_mask = __riscv_vmfgt_vf_f64m1_b64(ma, 0.0, mw);
        vbool64_t neg_mask = __riscv_vmflt_vf_f64m1_b64(ma, 0.0, mw);
        
        vfloat64m1_t mr = zero;
        mr = __riscv_vmerge_vvm_f64m1(mr, one, pos_mask, mw);
        mr = __riscv_vmerge_vvm_f64m1(mr, neg_one, neg_mask, mw);

        __riscv_vse64_v_f64m1(&res[i], mr, mw);
    }

__ort_math_backend_sign_double_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, double)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}