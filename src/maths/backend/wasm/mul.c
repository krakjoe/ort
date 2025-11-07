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

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, mul, int16_t) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;
    const size_t mw = 8; // 8 int16_t per 128-bit WASM SIMD register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_mul_int16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mb = wasm_v128_load(&vb[i]);
        v128_t mr = wasm_i16x8_mul(ma, mb);
        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_mul_int16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(mul, int16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, mul, int32_t) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;
    const size_t mw = 4; // 4 int32_t per 128-bit WASM SIMD register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_mul_int32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mb = wasm_v128_load(&vb[i]);
        v128_t mr = wasm_i32x4_mul(ma, mb);
        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_mul_int32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(mul, int32_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, mul, uint16_t) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;
    const size_t mw = 8; // 8 uint16_t per 128-bit WASM SIMD register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_mul_uint16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mb = wasm_v128_load(&vb[i]);
        v128_t mr = wasm_i16x8_mul(ma, mb);
        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_mul_uint16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(mul, uint16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, mul, uint32_t) {
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;
    const size_t mw = 4; // 4 uint32_t per 128-bit WASM SIMD register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_mul_uint32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mb = wasm_v128_load(&vb[i]);
        v128_t mr = wasm_i32x4_mul(ma, mb);
        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_mul_uint32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(mul, uint32_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, mul, float32) {
    const float32* va = (const float32*)a;
    const float32* vb = (const float32*)b;
    float32* res = (float32*)result;
    const size_t mw = 4; // 4 float32 per 128-bit WASM SIMD register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_mul_float32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mb = wasm_v128_load(&vb[i]);
        v128_t mr = wasm_f32x4_mul(ma, mb);
        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_mul_float32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(mul, float32)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}
