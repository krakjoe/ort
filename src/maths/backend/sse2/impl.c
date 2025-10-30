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

#include "maths/backend/sse2/impl.h"
#include "maths/backend/guard.h"

#include <emmintrin.h> /* SSE2 */

ORT_TLS ort_memcpy_func_t ort_memcpy_fallback = NULL;

static void* ort_math_memcpy_sse2(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t*)dest;
    const uint8_t *s =
      (const uint8_t*)src;

    if (n == 0) {
        goto __ort_math_memcpy_sse2_yield;
    }

    size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        __m128i chunk =
            _mm_load_si128(
            (const __m128i*)(s + i));

        _mm_store_si128(
            (__m128i*)(d + i), chunk);
    }

    ort_memcpy_fallback(
      d + i, s + i, n - i);

__ort_math_memcpy_sse2_yield:
    return dest;
}


void ort_math_backend_cpu_install(ort_math_dispatch_t* table) {
    /* don't clobber the environment with 
        features not supported on this core */
    if (ORT_MATH_BACKEND_GUARD(ORT_MATH_BACKEND_SSE2)) {
        return;
    }

    /* set allocation alignment to SSE2 vector length */
    ort_alloc_align(16);

    /* insert memcpy implementation */
    ort_memcpy_fallback =
        ort_alloc_memcpy(
            ort_math_memcpy_sse2);

    /* abs.c */
    ORT_MATH_BACKEND_INSTALL(table, sse2, FLOAT,   abs, float)
    ORT_MATH_BACKEND_INSTALL(table, sse2, DOUBLE,  abs, double)

    /* add.c */
    ORT_MATH_BACKEND_INSTALL(table, sse2, FLOAT,   add, float)
    ORT_MATH_BACKEND_INSTALL(table, sse2, DOUBLE,  add, double)
    ORT_MATH_BACKEND_INSTALL(table, sse2, INT8,    add, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, INT16,   add, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, INT32,   add, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, UINT8,   add, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, UINT16,  add, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, UINT32,  add, uint32_t)

    /* div.c */
    ORT_MATH_BACKEND_INSTALL(table, sse2, FLOAT,   div, float)
    ORT_MATH_BACKEND_INSTALL(table, sse2, DOUBLE,  div, double)

    /* matmul.c */
    ORT_MATH_BACKEND_INSTALL(table, sse2, FLOAT,   matmul, float)
    ORT_MATH_BACKEND_INSTALL(table, sse2, DOUBLE,  matmul, double)
    ORT_MATH_BACKEND_INSTALL(table, sse2, INT16,   matmul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, UINT16,  matmul, uint16_t)

    /* mul.c */
    ORT_MATH_BACKEND_INSTALL(table, sse2, FLOAT,   mul, float)
    ORT_MATH_BACKEND_INSTALL(table, sse2, DOUBLE,  mul, double)
    ORT_MATH_BACKEND_INSTALL(table, sse2, INT16,   mul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, UINT16,  mul, uint16_t)

    /* neg.c */
    ORT_MATH_BACKEND_INSTALL(table, sse2, FLOAT,   neg, float)
    ORT_MATH_BACKEND_INSTALL(table, sse2, DOUBLE,  neg, double)

    /* recip.c */
    ORT_MATH_BACKEND_INSTALL(table, sse2, FLOAT,   recip, float)
    ORT_MATH_BACKEND_INSTALL(table, sse2, DOUBLE,  recip, double)

    /* sign.c */
    ORT_MATH_BACKEND_INSTALL(table, sse2, FLOAT,   sign, float)
    ORT_MATH_BACKEND_INSTALL(table, sse2, DOUBLE,  sign, double)

    /* sqrt.c */
    ORT_MATH_BACKEND_INSTALL(table, sse2, FLOAT,   sqrt, float)
    ORT_MATH_BACKEND_INSTALL(table, sse2, DOUBLE,  sqrt, double)

    /* sub.c */
    ORT_MATH_BACKEND_INSTALL(table, sse2, FLOAT,   sub, float)
    ORT_MATH_BACKEND_INSTALL(table, sse2, DOUBLE,  sub, double)
    ORT_MATH_BACKEND_INSTALL(table, sse2, INT8,    sub, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, INT16,   sub, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, INT32,   sub, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, INT64,   sub, int64_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, UINT8,   sub, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, UINT16,  sub, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, sse2, UINT32,  sub, uint32_t)
  }
