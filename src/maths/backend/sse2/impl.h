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

ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, abs, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, abs, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, add, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, div, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, div, double);

ORT_MATH_BACKEND_MATMUL_OP_DECL(sse2, float);
ORT_MATH_BACKEND_MATMUL_OP_DECL(sse2, double);
ORT_MATH_BACKEND_MATMUL_OP_DECL(sse2, int16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(sse2, uint16_t);

ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, mul, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, mul, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, mul, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, mul, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, neg, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, neg, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, recip, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, recip, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, sign, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, sign, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, sqrt, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(sse2, sqrt, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(sse2, sub, double);