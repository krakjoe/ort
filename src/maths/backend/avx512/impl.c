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

#include "maths/backend/avx512/impl.h"
#include "maths/backend/guard.h"

#include <immintrin.h>  /* AVX512 */

#ifdef __AVX512F__

ORT_TLS ort_memcpy_func_t ort_memcpy_fallback = NULL;

static void* ort_math_memcpy_avx512(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t*)dest;
    const uint8_t *s =
      (const uint8_t*)src;

    if (n == 0) {
        goto __ort_math_memcpy_avx512_yield;
    }

    size_t i = 0;

    for (; i + 64 <= n; i += 64) {
        __m512i chunk =
            _mm512_load_si512(
            (const __m512i*)(s + i));
        
        _mm512_store_si512(
            (__m512i*)(d + i), chunk);
    }

    ort_memcpy_fallback(
      d + i, s + i, n - i);

__ort_math_memcpy_avx512_yield:
    return dest;
}

void ort_math_backend_cpu_install(ort_math_dispatch_t* table) {
    /* don't clobber the environment with 
        features not supported on this core */
    if (ORT_MATH_BACKEND_GUARD(ORT_MATH_BACKEND_AVX512)) {
        return;
    }

    /* set allocation alignment to AVX512 vector length */
    ort_alloc_align(64);

    /* insert memcpy implementation */
    ort_memcpy_fallback =
        ort_alloc_memcpy(
            ort_math_memcpy_avx512);

    /* abs.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   abs, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  abs, double)

    /* add.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   add, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  add, double)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT8,    add, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT16,   add, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT32,   add, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT64,   add, int64_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT8,   add, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT16,  add, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT32,  add, uint32_t)

    /* ceil.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   ceil, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  ceil, double)

    /* div.c*/
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   div, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  div, double)

    /* dot.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   dot, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  dot, double)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT16,   dot, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT32,   dot, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT16,  dot, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT32,  dot, uint32_t)

    /* floor.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   floor, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  floor, double)

    /* matmul.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   matmul, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  matmul, double)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT16,   matmul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT32,   matmul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT16,  matmul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT32,  matmul, uint32_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT8,    matmul, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT8,   matmul, uint8_t)

    /* mul.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   mul, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  mul, double)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT16,   mul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT32,   mul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT16,  mul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT32,  mul, uint32_t)

    /* neg.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   neg, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  neg, double)

    /* recip.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   recip, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  recip, double)

    /* round.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   round, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  round, double)

    /* sign.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   sign, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  sign, double)

    /* sqrt.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   sqrt, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  sqrt, double)

    /* sub.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   sub, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  sub, double)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT8,    sub, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT16,   sub, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT32,   sub, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, INT64,   sub, int64_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT8,   sub, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT16,  sub, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx512, UINT32,  sub, uint32_t)

    /* trunc.c */
    ORT_MATH_BACKEND_INSTALL(table, avx512, FLOAT,   trunc, float)
    ORT_MATH_BACKEND_INSTALL(table, avx512, DOUBLE,  trunc, double)
}

#endif