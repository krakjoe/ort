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

ORT_MATH_BACKEND_UTIL_DECL(
    hsum, float32x4, float, v128_t) {
    // WASM: sum all 4 lanes of a float32x4 vector
    float sum = wasm_f32x4_extract_lane(v, 0)
             + wasm_f32x4_extract_lane(v, 1)
             + wasm_f32x4_extract_lane(v, 2)
             + wasm_f32x4_extract_lane(v, 3);
    return sum;
}

ORT_MATH_BACKEND_UTIL_DECL(
    hsum, float64x2, double, v128_t) {
    // WASM: sum both lanes of a float64x2 vector
    return wasm_f64x2_extract_lane(v, 0) + wasm_f64x2_extract_lane(v, 1);
}

ORT_MATH_BACKEND_UTIL_DECL(
    hsum, int32x4, int32_t, v128_t) {
    return wasm_i32x4_extract_lane(v, 0)
         + wasm_i32x4_extract_lane(v, 1)
         + wasm_i32x4_extract_lane(v, 2)
         + wasm_i32x4_extract_lane(v, 3);
}

ORT_MATH_BACKEND_UTIL_DECL(
    hsum, uint32x4, uint32_t, v128_t) {
    return wasm_u32x4_extract_lane(v, 0)
         + wasm_u32x4_extract_lane(v, 1)
         + wasm_u32x4_extract_lane(v, 2)
         + wasm_u32x4_extract_lane(v, 3);
}
