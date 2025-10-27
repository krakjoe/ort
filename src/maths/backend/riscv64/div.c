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

#include <riscv_vector.h> /* RVV */

ORT_MATH_BACKEND_BINARY_OP_DECL(div, float) {
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    float* res = (float*)result;
    const size_t mw = __riscv_vsetvlmax_e32m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_div_float_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        vfloat32m1_t ma = __riscv_vle32_v_f32m1(&va[i], mw);
        vfloat32m1_t mb = __riscv_vle32_v_f32m1(&vb[i], mw);
        vfloat32m1_t mr = __riscv_vfdiv_vv_f32m1(ma, mb, mw);
        __riscv_vse32_v_f32m1(&res[i], mr, mw);
    }

__ort_math_backend_div_float_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(div, float)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(div, double) {
    const double* va = (const double*)a;
    const double* vb = (const double*)b;
    double* res = (double*)result;
    const size_t mw = __riscv_vsetvlmax_e64m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_div_double_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        vfloat64m1_t ma = __riscv_vle64_v_f64m1(&va[i], mw);
        vfloat64m1_t mb = __riscv_vle64_v_f64m1(&vb[i], mw);
        vfloat64m1_t mr = __riscv_vfdiv_vv_f64m1(ma, mb, mw);
        __riscv_vse64_v_f64m1(&res[i], mr, mw);
    }

__ort_math_backend_div_double_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(div, double)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}