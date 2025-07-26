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

ORT_MATH_BACKEND_BINARY_OP_DECL(div, float) {
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    float* res      = (float*)result;
    const size_t mw = 4; /* 4 float per 128-bit WASM SIMD register */

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_div_float_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t va4 = wasm_v128_load(&va[i]);
        v128_t vb4 = wasm_v128_load(&vb[i]);
        v128_t vr4 = wasm_f32x4_div(va4, vb4);
        wasm_v128_store(&res[i], vr4);
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