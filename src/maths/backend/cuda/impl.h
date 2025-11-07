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

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, abs, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, abs, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, ceil, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, ceil, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, div, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, div, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, dot, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, dot, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, dot, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, dot, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, dot, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, dot, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, floor, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, floor, float64);

ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, int8_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, uint8_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, int16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, uint16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, int32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, uint32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, float32);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, neg, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, neg, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, recip, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, recip, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, round, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, round, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sign, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sign, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sqrt, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sqrt, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, trunc, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, trunc, float64);

extern ORT_TLS ort_alloc_t          __ort_math_cpu_allocator;
extern ORT_TLS ort_math_dispatch_t* __ort_math_cpu_dispatch;
extern int                          __ort_cuda_threshold;
extern int                          __ort_cuda_threads;

void ort_cuda_activate(void);