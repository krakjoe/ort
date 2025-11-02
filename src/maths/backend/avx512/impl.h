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

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, abs, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, abs, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, add, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, add, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, add, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, add, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, add, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, add, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, add, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, add, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, add, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, ceil, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, ceil, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, div, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, div, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, dot, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, floor, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, floor, double);

ORT_MATH_BACKEND_MATMUL_OP_DECL(avx512, int8_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx512, uint8_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx512, int16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx512, uint16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx512, int32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx512, uint32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx512, float);
ORT_MATH_BACKEND_MATMUL_OP_DECL(avx512, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, mul, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, mul, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, mul, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, mul, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, mul, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, mul, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, neg, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, neg, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, recip, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, recip, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, round, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, round, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, sign, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, sign, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, sqrt, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, sqrt, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(avx512, sub, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, trunc, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(avx512, trunc, double);