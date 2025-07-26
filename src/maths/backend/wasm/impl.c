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

ORT_TLS ort_memcpy_func_t ort_memcpy_fallback = NULL;

static void* ort_math_memcpy_wasm(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t*)dest;
    const uint8_t *s = (const uint8_t*)src;

    if (n == 0) {
        return dest;
    }

    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        v128_t chunk =
            wasm_v128_load(&s[i]);
        wasm_v128_store(&d[i], chunk);
    }

    // Fallback for remaining bytes
    if (i < n) {
        ort_memcpy_fallback(d + i, s + i, n - i);
    }

    return dest;
}

void ort_math_backend_install(ort_math_dispatch_t* table) {
    /* There is no need to guard the wasm backend, there is
        no variability in the ISA */

    ort_alloc_align(16);

    ort_memcpy_fallback =
        ort_alloc_memcpy(
            ort_math_memcpy_wasm);

    /* abs.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   abs, float)
    ORT_MATH_BACKEND_INSTALL(table, INT8,    abs, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, INT16,   abs, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, INT32,   abs, int32_t)

    /* add.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   add, float)
    ORT_MATH_BACKEND_INSTALL(table, INT8,    add, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, INT16,   add, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, INT32,   add, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT8,   add, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT16,  add, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT32,  add, uint32_t)

    /* ceil.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   ceil,  float)

    /* div.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   div,   float)

    /* floor.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   floor, float)

    /* matmul.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   matmul, float)
    ORT_MATH_BACKEND_INSTALL(table, INT32,   matmul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT32,  matmul, uint32_t)

    /* mul.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   mul, float)
    ORT_MATH_BACKEND_INSTALL(table, INT16,   mul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, INT32,   mul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT16,  mul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT32,  mul, uint32_t)

    /* neg.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   neg, float)
    ORT_MATH_BACKEND_INSTALL(table, INT8,    neg, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, INT16,   neg, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, INT32,   neg, int32_t)

    /* recip.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,  recip, float)

    /* round.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,  round, float)

    /* sign.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,  sign,  float)

    /* sqrt.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,  sqrt,  float)

    /* sub.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,  sub,   float)
    ORT_MATH_BACKEND_INSTALL(table, INT8,   sub,   int8_t)
    ORT_MATH_BACKEND_INSTALL(table, INT16,  sub,   int16_t)
    ORT_MATH_BACKEND_INSTALL(table, INT32,  sub,   int32_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT8,  sub,   uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT16, sub,   uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT32, sub,   uint32_t)

    /* trunc.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,  trunc, float)
}