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

ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, abs, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, abs, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, div, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, div, float64);

ORT_MATH_BACKEND_MATMUL_OP_DECL(sse2, float32);
ORT_MATH_BACKEND_MATMUL_OP_DECL(sse2, float64);
ORT_MATH_BACKEND_MATMUL_OP_DECL(sse2, int16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(sse2, uint16_t);

ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, mul, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, mul, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, mul, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, mul, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, neg, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, neg, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, recip, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, recip, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, sign, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, sign, float64);

ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, sqrt, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, sqrt, float64);

ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, float64);