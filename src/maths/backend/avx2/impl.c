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
#include "maths/backend/guard.h"

#include <immintrin.h>  /* AVX/AVX2 */

ORT_TLS ort_memcpy_func_t ort_memcpy_fallback = NULL;

static void* ort_math_memcpy_avx2(void *dest, const void *src, size_t n) {
    uint8_t *d = (uint8_t*)dest;
    const uint8_t *s =
      (const uint8_t*)src;

    if (n == 0) {
        goto __ort_math_memcpy_avx2_yield;
    }

    size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256i chunk =
            _mm256_load_si256(
            (const __m256i*)(s + i));
        
        _mm256_store_si256(
            (__m256i*)(d + i), chunk);
    }

    ort_memcpy_fallback(
      d + i, s + i, n - i);

__ort_math_memcpy_avx2_yield:
    return dest;
}

void ort_math_backend_install(ort_math_dispatch_t* table) {
    /* don't clobber the environment with 
        features not supported on this core */
    if (ORT_MATH_BACKEND_GUARD(ORT_MATH_BACKEND_AVX2)) {
        return;
    }

    /* set allocation alignment to AVX2 vector length */
    ort_alloc_align(32);

    /* insert memcpy implementation */
    ort_memcpy_fallback =
        ort_alloc_memcpy(
            ort_math_memcpy_avx2);

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

    /* ceil.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   ceil, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  ceil, double)

    /* div.c*/
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   div, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  div, double)

    /* floor.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   floor, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  floor, double)

    /* matmul.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   matmul, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  matmul, double)
    ORT_MATH_BACKEND_INSTALL(table, INT16,   matmul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, INT32,   matmul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT16,  matmul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT32,  matmul, uint32_t)
    ORT_MATH_BACKEND_INSTALL(table, INT8,    matmul, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT8,   matmul, uint8_t)

    /* mul.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   mul, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  mul, double)
    ORT_MATH_BACKEND_INSTALL(table, INT16,   mul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, INT32,   mul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT16,  mul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, UINT32,  mul, uint32_t)

    /* neg.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   neg, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  neg, double)

    /* recip.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   recip, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  recip, double)

    /* round.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   round, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  round, double)

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

    /* trunc.c */
    ORT_MATH_BACKEND_INSTALL(table, FLOAT,   trunc, float)
    ORT_MATH_BACKEND_INSTALL(table, DOUBLE,  trunc, double)
}