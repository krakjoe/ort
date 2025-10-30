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

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, int8_t) {
    const int8_t* va = (const int8_t*)a;
    const int8_t* vb = (const int8_t*)b;
    int8_t* res = (int8_t*)result;
    const size_t mw = 16; // 16 int8_t per 128-bit WASM SIMD register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_int8_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mb = wasm_v128_load(&vb[i]);
        v128_t mr = wasm_i8x16_add(ma, mb);
        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_add_int8_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, int8_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, int16_t) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;
    const size_t mw = 8; // 8 int16_t per 128-bit WASM SIMD register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_int16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mb = wasm_v128_load(&vb[i]);
        v128_t mr = wasm_i16x8_add(ma, mb);
        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_add_int16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, int16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, int32_t) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;
    const size_t mw = 4; // 4 int32_t per 128-bit WASM SIMD register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_int32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mb = wasm_v128_load(&vb[i]);
        v128_t mr = wasm_i32x4_add(ma, mb);
        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_add_int32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, int32_t)(
            res    + mc,
            va     + mc,
            vb     + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, uint8_t) {
    const uint8_t* va = (const uint8_t*)a;
    const uint8_t* vb = (const uint8_t*)b;
    uint8_t* res = (uint8_t*)result;
    const size_t mw = 16; // 16 uint8_t per 128-bit WASM SIMD register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_uint8_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mb = wasm_v128_load(&vb[i]);
        v128_t mr = wasm_i8x16_add(ma, mb);
        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_add_uint8_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, uint8_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, uint16_t) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;
    const size_t mw = 8; // 8 uint16_t per 128-bit WASM SIMD register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_uint16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mb = wasm_v128_load(&vb[i]);
        v128_t mr = wasm_i16x8_add(ma, mb);
        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_add_uint16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, uint16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, uint32_t) {
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;
    const size_t mw = 4; // 4 uint32_t per 128-bit WASM SIMD register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_uint32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mb = wasm_v128_load(&vb[i]);
        v128_t mr = wasm_i32x4_add(ma, mb);
        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_add_uint32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, uint32_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, float) {
    const float* va = (const float*)a;
    const float* vb = (const float*)b;
    float* res = (float*)result;
    const size_t mw = 4; // 4 float per 128-bit WASM SIMD register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_float_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        v128_t ma = wasm_v128_load(&va[i]);
        v128_t mb = wasm_v128_load(&vb[i]);
        v128_t mr = wasm_f32x4_add(ma, mb);
        wasm_v128_store(&res[i], mr);
    }

__ort_math_backend_add_float_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, float)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}