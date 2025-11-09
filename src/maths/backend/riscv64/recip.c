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

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, recip, float16) {
    const float16* va = (const float16*)a;
    float16* res = (float16*)result;
    const size_t mw = __riscv_vsetvlmax_e16m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_recip_float16_fallback;
    }

    const vfloat16m1_t one = __riscv_vfmv_v_f_f16m1(1.0f, mw);

    for (size_t i = 0; i < mc; i += mw) {
        vfloat16m1_t ma =
            __riscv_vle16_v_f16m1(
                (const _Float16*)&va[i], mw);
        vfloat16m1_t mr = __riscv_vfdiv_vv_f16m1(one, ma, mw);
        __riscv_vse16_v_f16m1(
            (_Float16*)&res[i], mr, mw);
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

ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, recip, float32) {
    const float32* va = (const float32*)a;
    float32* res = (float32*)result;
    const size_t mw = __riscv_vsetvlmax_e32m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_recip_float32_fallback;
    }

    const vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, mw);

    /* Vectorized loop */
    for (size_t i = 0; i < mc; i += mw) {
        vfloat32m1_t ma = __riscv_vle32_v_f32m1(&va[i], mw);
        vfloat32m1_t mr = __riscv_vfdiv_vv_f32m1(one, ma, mw);
        __riscv_vse32_v_f32m1(&res[i], mr, mw);
    }

__ort_math_backend_recip_float32_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(recip, float32)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, recip, float64) {
    const float64* va = (const float64*)a;
    float64* res = (float64*)result;
    const size_t mw = __riscv_vsetvlmax_e64m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        /* Not enough elements for a single SIMD operation, fallback to scalar */
        goto __ort_math_backend_recip_float64_fallback;
    }

    const vfloat64m1_t one = __riscv_vfmv_v_f_f64m1(1.0, mw);

    /* Vectorized loop */
    for (size_t i = 0; i < mc; i += mw) {
        vfloat64m1_t ma = __riscv_vle64_v_f64m1(&va[i], mw);
        vfloat64m1_t mr = __riscv_vfdiv_vv_f64m1(one, ma, mw);
        __riscv_vse64_v_f64m1(&res[i], mr, mw);
    }

__ort_math_backend_recip_float64_fallback:
    /* Handle remaining elements with scalar operations */
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(recip, float64)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}