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

ORT_TLS ort_memcpy_func_t ort_memcpy_fallback = NULL;

static void* ort_math_memcpy_riscv64(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t*)dest;
    const uint8_t *s =
      (const uint8_t*)src;

    if (n == 0) {
        goto __ort_math_memcpy_riscv64_yield;
    }

    size_t i = 0;
    size_t mw;

    for (; i < n; i += mw) {
        mw = __riscv_vsetvl_e8m1(n - i);
        vuint8m1_t chunk =
            __riscv_vle8_v_u8m1(s + i, mw);

        __riscv_vse8_v_u8m1(d + i, chunk, mw);
    }

__ort_math_memcpy_riscv64_yield:
    return dest;
}

void ort_math_backend_install(ort_math_dispatch_t* table) {
    /* There is no need to guard the riscv64 backend, there is
        no variability in the ISA */

    /* set allocation alignment to RVV vector length */
    ort_alloc_align(16);

    /* insert memcpy implementation */
    ort_memcpy_fallback =
        ort_alloc_memcpy(
            ort_math_memcpy_riscv64);

    /* abs.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   abs, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  abs, double)

    /* add.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   add, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  add, double)
    ORT_MATH_BACKEND_INSTALL(table, INT8,    add, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, INT16,   add, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, INT32,   add, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, INT64,   add, int64_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT8,   add, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT16,  add, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT32,  add, uint32_t)

    /* div.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   div, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  div, double)

    /* matmul.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   matmul, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  matmul, double)
    ORT_MATH_BACKEND_INSTALL(table, INT16,   matmul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT16,  matmul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, INT32,   matmul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT32,  matmul, uint32_t)

    /* mul.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   mul, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  mul, double)
    ORT_MATH_BACKEND_INSTALL(table, INT8,    mul, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT8,   mul, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, INT16,   mul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT16,  mul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, INT32,   mul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, INT64,   mul, int64_t)

    /* neg.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   neg, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  neg, double)

    /* recip.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   recip, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  recip, double)

    /* sign.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   sign, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  sign, double)

    /* sqrt.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   sqrt, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  sqrt, double)

    /* sub.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   sub, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  sub, double)
    ORT_MATH_BACKEND_INSTALL(table, INT8,    sub, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, INT16,   sub, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, INT32,   sub, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, INT64,   sub, int64_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT8,   sub, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT16,  sub, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT32,  sub, uint32_t)
}