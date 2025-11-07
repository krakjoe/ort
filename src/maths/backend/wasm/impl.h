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

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, abs, int8_t);
ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, abs, int16_t);
ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, abs, int32_t);
ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, abs, float32);

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, add, float32);

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, ceil, float32);

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, div, float32);

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, floor, float32);

ORT_MATH_BACKEND_MATMUL_OP_DECL(wasm, float32);
ORT_MATH_BACKEND_MATMUL_OP_DECL(wasm, int32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(wasm, uint32_t);

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, mul, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, mul, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, mul, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, mul, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, mul, float32);

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, neg, int8_t);
ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, neg, int16_t);
ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, neg, int32_t);
ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, neg, float32);

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, recip, float32);

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, round, float32);

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, sign, float32);

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, sqrt, float32);

ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, sub, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, sub, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, sub, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, sub, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, sub, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, sub, uint32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(wasm, sub, float32);

ORT_MATH_BACKEND_UNARY_OP_DECL(wasm, trunc, float32);