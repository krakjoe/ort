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

ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, abs, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, abs, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, div, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, div, double);

ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, int16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, uint16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, int32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, uint32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, float);
ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, neg, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, neg, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, recip, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, recip, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, sign, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, sign, double);

ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, sqrt, float);
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, sqrt, double);

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, float);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, double);