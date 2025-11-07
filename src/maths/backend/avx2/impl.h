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

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, abs, float16);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, abs, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, abs, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, float16);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, add, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, ceil, float16);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, ceil, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, ceil, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, div, float16);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, div, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, div, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, float16);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, dot, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, floor, float16);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, floor, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, floor, float64);

ORT_MATH_BACKEND_MATMUL_OP_DECL(avx2, int8_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx2, uint8_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx2, int16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx2, uint16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx2, int32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx2, uint32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx2, float16);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx2, float32);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx2, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, mul, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, mul, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, mul, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, mul, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, mul, float16);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, mul, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, mul, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, neg, float16);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, neg, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, neg, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, recip, float16);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, recip, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, recip, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, round, float16);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, round, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, round, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, sign, float16);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, sign, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, sign, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, sqrt, float16);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, sqrt, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, sqrt, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, sub, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, sub, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, sub, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, sub, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, sub, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, sub, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, sub, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, sub, float16);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, sub, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx2, sub, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, trunc, float16);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, trunc, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx2, trunc, float64);