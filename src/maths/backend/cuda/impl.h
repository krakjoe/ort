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

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, abs, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, abs, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, add, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, ceil, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, ceil, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, div, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, div, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, dot, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, dot, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, dot, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, dot, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, dot, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, dot, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, floor, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, floor, double);

ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, int8_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, uint8_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, int16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, uint16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, int32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, uint32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, float);
ORT_MATH_BACKEND_MATMUL_OP_DECL(cuda, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, mul, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, neg, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, neg, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, recip, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, recip, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, round, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, round, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sign, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sign, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sqrt, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, sqrt, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(cuda, sub, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, trunc, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(cuda, trunc, double);

extern ORT_TLS ort_alloc_t          __ort_math_cpu_allocator;
extern ORT_TLS ort_math_dispatch_t* __ort_math_cpu_dispatch;
extern int                          __ort_cuda_threshold;

void ort_cuda_activate(void);