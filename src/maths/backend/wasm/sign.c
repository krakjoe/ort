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

#include <wasm_simd128.h>  /* WASM */

ORT_MATH_BACKEND_UNARY_OP_DECL(sign, float) {
    const float* va = (const float*)a;
    float* res      = (float*)result;
    const size_t mw = 4; // 4 floats per 128-bit WASM SIMD
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sign_float_fallback;
    }

    // Vectorized loop - process 4 floats at once using WASM SIMD
    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t zero = wasm_f32x4_splat(0.0f);
        v128_t one = wasm_f32x4_splat(1.0f);
        v128_t neg_one = wasm_f32x4_splat(-1.0f);

        // Create masks for positive and negative values
        v128_t pos_mask = wasm_f32x4_gt(ma, zero); // ma > 0
        v128_t neg_mask = wasm_f32x4_lt(ma, zero); // ma < 0

        // result = (ma > 0) ? 1.0f : 0.0f
        v128_t result = wasm_v128_and(pos_mask, one);
        // mr = (ma < 0) ? -1.0f : result
        v128_t neg_val = wasm_v128_and(neg_mask, neg_one);
        v128_t mr = wasm_v128_or(neg_val, result);

        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_sign_float_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sign, float)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
