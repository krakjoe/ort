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

#include "maths/backend/riscv64/impl.h"

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

void ort_math_backend_cpu_install(ort_math_dispatch_t* table) {
    /* There is no need to guard the riscv64 backend, there is
        no variability in the ISA */

    /* set allocation alignment to RVV vector length */
    ort_alloc_align(16);

    /* insert memcpy implementation */
    ort_memcpy_fallback =
        ort_alloc_memcpy(
            ort_math_memcpy_riscv64);

    /* abs.c */
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT32,   abs, float32)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT64,  abs, float64)

    /* add.c */
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT32,   add, float32)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT64,  add, float64)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT8,    add, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT16,   add, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT32,   add, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT64,   add, int64_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, UINT8,   add, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, UINT16,  add, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, UINT32,  add, uint32_t)

    /* div.c */
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT32,   div, float32)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT64,  div, float64)

    /* matmul.c */
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT32,   matmul, float32)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT64,  matmul, float64)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT16,   matmul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, UINT16,  matmul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT32,   matmul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, UINT32,  matmul, uint32_t)

    /* mul.c */
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT32,   mul, float32)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT64,  mul, float64)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT8,    mul, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, UINT8,   mul, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT16,   mul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, UINT16,  mul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT32,   mul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT64,   mul, int64_t)

    /* neg.c */
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT32,   neg, float32)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT64,  neg, float64)

    /* recip.c */
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT32,   recip, float32)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT64,  recip, float64)

    /* sign.c */
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT32,   sign, float32)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT64,  sign, float64)

    /* sqrt.c */
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT32,   sqrt, float32)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT64,  sqrt, float64)

    /* sub.c */
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT32,   sub, float32)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, FLOAT64,  sub, float64)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT8,    sub, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT16,   sub, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT32,   sub, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, INT64,   sub, int64_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, UINT8,   sub, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, UINT16,  sub, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, riscv64, UINT32,  sub, uint32_t)
}