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

#include "maths/backend/wasm/impl.h"
#include "maths/backend/wasm/util.h"

#include <wasm_simd128.h>  /* WASM */

ORT_MATH_BACKEND_MATMUL_OP_DECL(wasm, float) {
    const float *va = (const float *)a;
    const float *vb = (const float *)b;
    float *res = (float *)result;
    for (size_t j = 0; j < b_cols; j++) {
        float sum = 0.0f;
        const size_t mw = 4;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            v128_t ma = wasm_v128_load(&va[k]);
            float mb_arr[4] = {
                vb[k * b_cols + j],
                vb[(k+1) * b_cols + j],
                vb[(k+2) * b_cols + j],
                vb[(k+3) * b_cols + j]
            };
            v128_t mb = wasm_v128_load(mb_arr);
            v128_t mr = wasm_f32x4_mul(ma, mb);
            sum += ORT_MATH_BACKEND_UTIL(wasm, hsum, float32x4, float)(mr);
        }
        for (; k < a_cols; k++) {
            sum += va[k] * vb[k * b_cols + j];
        }
        res[j] = sum;
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(wasm, int32_t) {
    const int32_t *va = (const int32_t *)a;
    const int32_t *vb = (const int32_t *)b;
    int32_t *res = (int32_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        int64_t sum = 0;
        const size_t mw = 4;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            v128_t ma = wasm_v128_load(&va[k]);
            int32_t mb_arr[4] = {
                vb[k * b_cols + j],
                vb[(k+1) * b_cols + j],
                vb[(k+2) * b_cols + j],
                vb[(k+3) * b_cols + j]
            };
            v128_t mb = wasm_v128_load(mb_arr);
            v128_t mr = wasm_i32x4_mul(ma, mb);
            sum += ORT_MATH_BACKEND_UTIL(wasm, hsum, int32x4, int32_t)(mr);
        }
        for (; k < a_cols; k++) {
            sum += va[k] * vb[k * b_cols + j];
        }
        res[j] = ORT_MATH_CLAMP(sum, INT32, int32_t);
    }
}

ORT_MATH_BACKEND_MATMUL_OP_DECL(wasm, uint32_t) {
    const uint32_t *va = (const uint32_t *)a;
    const uint32_t *vb = (const uint32_t *)b;
    uint32_t *res = (uint32_t *)result;
    for (size_t j = 0; j < b_cols; j++) {
        int64_t sum = 0;
        const size_t mw = 4;
        size_t mc = ort_math_backend_optimal_count(a_cols, mw);
        size_t k = 0;
        for (; k < mc; k += mw) {
            v128_t ma = wasm_v128_load(&va[k]);
            uint32_t mb_arr[4] = {
                vb[k * b_cols + j],
                vb[(k+1) * b_cols + j],
                vb[(k+2) * b_cols + j],
                vb[(k+3) * b_cols + j]
            };
            v128_t mb = wasm_v128_load(mb_arr);
            /* there's no unsigned equivalent, this is correct, not a mistake */
            v128_t mr = wasm_i32x4_mul(ma, mb);
            sum += ORT_MATH_BACKEND_UTIL(wasm, hsum, uint32x4, uint32_t)(mr);
        }
        for (; k < a_cols; k++) {
            sum += va[k] * vb[k * b_cols + j];
        }
        res[j] = ORT_MATH_CLAMP_MAX(sum, UINT32, uint32_t);
    }
}