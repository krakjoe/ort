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

#include <wasm_simd128.h>  /* WASM */

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, sqrt, float32) {
    const float32* va = (const float32*)a;
    float32* res      = (float32*)result;
    const size_t mw = 4; // 4 float32 per 128-bit WASM SIMD
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sqrt_float32_fallback;
    }

    // Vectorized loop - process 4 float32 at once using WASM SIMD
    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mr = wasm_f32x4_sqrt(ma);
        wasm_v128_store(&res[i], mr);
    }

    if (mc < count) {
__ort_math_backend_sqrt_float32_fallback:
        ORT_MATH_FRONTEND_OP_SYMBOL(sqrt, float32)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}
