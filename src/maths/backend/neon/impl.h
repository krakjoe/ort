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
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, abs, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, abs, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, abs, float64);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, abs, int8_t);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, abs, int16_t);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, abs, int32_t);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, abs, int64_t);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, float16);
#endif
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, float64);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, uint32_t);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, ceil, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, ceil, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, ceil, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, div, float16);
#endif
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, div, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, div, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, floor, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, floor, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, floor, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, float16);
#endif
ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, float32);
ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, float64);
ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, int8_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, uint8_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, int16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, uint16_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, int32_t);
ORT_MATH_BACKEND_MATMUL_OP_DECL(neon, uint32_t);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, mul, float16);
#endif
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, mul, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, mul, float64);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, mul, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, mul, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, mul, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, mul, uint32_t);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, float64);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, int8_t);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, int16_t);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, int32_t);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, neg, int64_t);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, recip, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, recip, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, recip, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, round, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, round, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, round, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, sign, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, sign, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, sign, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, sqrt, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, sqrt, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, sqrt, float64);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, sub, float16);
#endif
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, sub, float32);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, sub, float64);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, sub, int8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, sub, int16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, sub, int32_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, sub, int64_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, sub, uint8_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, sub, uint16_t);
ORT_MATH_BACKEND_BINARY_OP_DECL(neon, sub, uint32_t);

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, trunc, float16);
#endif
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, trunc, float32);
ORT_MATH_BACKEND_UNARY_OP_DECL(neon, trunc, float64);