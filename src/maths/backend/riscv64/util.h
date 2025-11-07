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

ORT_MATH_BACKEND_UTIL_DECL_VAR(riscv64, 
    hsum, float32xN, float32, vfloat32m1_t) {
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t sum_vec = __riscv_vfredosum_vs_f32m1_f32m1(v, zero, vl);
    return __riscv_vfmv_f_s_f32m1_f32(sum_vec);
}

ORT_MATH_BACKEND_UTIL_DECL_VAR(riscv64, 
    hsum, float64xN, float64, vfloat64m1_t) {
    vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    vfloat64m1_t sum_vec = __riscv_vfredosum_vs_f64m1_f64m1(v, zero, vl);
    return __riscv_vfmv_f_s_f64m1_f64(sum_vec);
}

ORT_MATH_BACKEND_UTIL_DECL_VAR(riscv64, 
    hsum, uint32xN, uint32_t, vuint32m1_t) {
    vuint32m1_t zero = __riscv_vmv_v_x_u32m1(0, 1);
    vuint32m1_t sum_vec = __riscv_vredsum_vs_u32m1_u32m1(v, zero, vl);
    return __riscv_vmv_x_s_u32m1_u32(sum_vec);
}

ORT_MATH_BACKEND_UTIL_DECL_VAR(riscv64, 
    hsum, int32xN, int32_t, vint32m1_t) {
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t sum_vec = __riscv_vredsum_vs_i32m1_i32m1(v, zero, vl);
    return __riscv_vmv_x_s_i32m1_i32(sum_vec);
}

ORT_MATH_BACKEND_UTIL_DECL_VAR(riscv64, 
    hsum, int16xN, int32_t, vint32m1_t) {
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t sum_vec = __riscv_vredsum_vs_i32m1_i32m1(v, zero, vl);
    return __riscv_vmv_x_s_i32m1_i32(sum_vec);
}