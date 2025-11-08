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

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, abs, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, abs, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, abs, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, float16);
#endif
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, float64);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, add, uint32_t);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, div, float16);
#endif
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, div, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, div, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, float16);
#endif
ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, float32);
ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, float64);
ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, int16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, uint16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, int32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(riscv64, uint32_t);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, float16);
#endif
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, float64);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, mul, uint32_t);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, neg, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, neg, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, neg, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, recip, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, recip, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, recip, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, sign, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, sign, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, sign, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, sqrt, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, sqrt, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(riscv64, sqrt, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, float16);
#endif
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, float64);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, uint32_t);
