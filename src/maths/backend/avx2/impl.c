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

#include "maths/backend/avx2/impl.h"
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

void ort_math_backend_cpu_install(ort_math_dispatch_t* table) {
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
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, abs, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, abs, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, abs, float64)

    /* add.c */
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, add, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, add, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, add, float64)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT8,    add, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT16,   add, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT32,   add, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT64,   add, int64_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT8,   add, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT16,  add, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT32,  add, uint32_t)

    /* ceil.c */
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, ceil, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, ceil, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, ceil, float64)

    /* div.c*/
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, div, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, div, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, div, float64)

    /* dot.c */
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, dot, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, dot, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, dot, float64)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT16,   dot, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT32,   dot, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT16,  dot, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT32,  dot, uint32_t)

    /* floor.c */
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, floor, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, floor, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, floor, float64)

    /* matmul.c */
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, matmul, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, matmul, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, matmul, float64)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT16,   matmul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT32,   matmul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT16,  matmul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT32,  matmul, uint32_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT8,    matmul, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT8,   matmul, uint8_t)

    /* mul.c */
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, mul, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, mul, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, mul, float64)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT16,   mul, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT32,   mul, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT16,  mul, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT32,  mul, uint32_t)

    /* neg.c */
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, neg, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, neg, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, neg, float64)

    /* recip.c */
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, recip, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, recip, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, recip, float64)

    /* round.c */
#ifdef ORT_BACKEND_CPU_F16C 
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, round, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, round, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, round, float64)

    /* sign.c */
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, sign, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, sign, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, sign, float64)

    /* sqrt.c */
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, sqrt, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, sqrt, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, sqrt, float64)

    /* sub.c */
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, sub, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, sub, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, sub, float64)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT8,    sub, int8_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT16,   sub, int16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT32,   sub, int32_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, INT64,   sub, int64_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT8,   sub, uint8_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT16,  sub, uint16_t)
    ORT_MATH_BACKEND_INSTALL(table, avx2, UINT32,  sub, uint32_t)

    /* trunc.c */
#ifdef ORT_BACKEND_CPU_F16C
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT16, trunc, float16)
#endif
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT32, trunc, float32)
    ORT_MATH_BACKEND_INSTALL(table, avx2, FLOAT64, trunc, float64)
}