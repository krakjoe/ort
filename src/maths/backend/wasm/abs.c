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

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, abs, int8_t) {
    const int8_t* va = (const int8_t*)a;
    int8_t* res      = (int8_t*)result;
    const size_t mw = 16; // 16 int8_t per 128-bit WASM SIMD register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_abs_int8_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mr = wasm_i8x16_abs(ma);
        wasm_v128_store(&res[i], mr);
    }

    if (mc < count) {
__ort_math_backend_abs_int8_fallback:
        ORT_MATH_FRONTEND_OP_SYMBOL(abs, int8_t)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, abs, int16_t) {
    const int16_t* va = (const int16_t*)a;
    int16_t* res      = (int16_t*)result;
    const size_t mw = 8; // 8 int16_t per 128-bit WASM SIMD register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_abs_int16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mr = wasm_i16x8_abs(ma);
        wasm_v128_store(&res[i], mr);
    }

    if (mc < count) {
__ort_math_backend_abs_int16_fallback:
        ORT_MATH_FRONTEND_OP_SYMBOL(abs, int16_t)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, abs, int32_t) {
    const int32_t* va = (const int32_t*)a;
    int32_t* res      = (int32_t*)result;
    const size_t mw = 4; // 4 int32_t per 128-bit WASM SIMD register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_abs_int32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mr = wasm_i32x4_abs(ma);
        wasm_v128_store(&res[i], mr);
    }

    if (mc < count) {
__ort_math_backend_abs_int32_fallback:
        ORT_MATH_FRONTEND_OP_SYMBOL(abs, int32_t)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, abs, float32) {
    const float32* va = (const float32*)a;
    float32* res      = (float32*)result;
    const size_t mw = 4; // 4 float32 per 128-bit WASM SIMD register
    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_abs_float32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mr = wasm_f32x4_abs(ma);
        wasm_v128_store(&res[i], mr);
    }

    if (mc < count) {
__ort_math_backend_abs_float32_fallback:
        ORT_MATH_FRONTEND_OP_SYMBOL(abs, float32)(
            res   + mc,
            va    + mc,
            count - mc);
    }
}