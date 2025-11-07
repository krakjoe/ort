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

/*
 @brief Implements frontend mul operations for tensors
 @test tests/math/mul
*/

#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/schema/mul.h"

ORT_MATH_FRONTEND_BINARY_OP_DECL(mul, float16) {
    float16* res = (float16*)result;
    const float16* va = (const float16*)a;
    const float16* vb = (const float16*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = ort_math_float16_from_float32(
            ort_math_float32_from_float16(va[i]) *
            ort_math_float32_from_float16(vb[i])
        );
    }
}

#define ORT_MATH_FRONTEND_MUL_IMPL(c_type, onnx_type) \
    ORT_MATH_FRONTEND_BINARY_OP_IMPL(mul, c_type, *)
ORT_MATH_FOREACH_NUMERIC_TYPE(
    ORT_MATH_FRONTEND_MUL_IMPL)
#undef ORT_MATH_FRONTEND_MUL_IMPL
ORT_MATH_FRONTEND_BINARY_OP_IMPL(mul, zend_bool, &&)

ORT_MATH_FRONTEND_SCALAR_OP_DECL(mul, float16) {
    float16* res = (float16*)result;
    const float16* va = (const float16*)a;
    const float16 sb = *(const float16*)b;
    for (size_t i = 0; i < count; i++) {
        res[i] = ort_math_float16_from_float32(
            ort_math_float32_from_float16(va[i]) *
            ort_math_float32_from_float16(sb)
        );
    }
}

#define ORT_MATH_FRONTEND_MUL_SCALAR_IMPL(c_type, onnx_type) \
    ORT_MATH_FRONTEND_SCALAR_OP_IMPL(mul, c_type, *)
ORT_MATH_FOREACH_NUMERIC_TYPE(
    ORT_MATH_FRONTEND_MUL_SCALAR_IMPL)
#undef ORT_MATH_FRONTEND_MUL_SCALAR_IMPL
ORT_MATH_FRONTEND_SCALAR_OP_IMPL(mul, zend_bool, &&)

ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(
    ort_math_kernel_binary_t, mul)
ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(
    ort_math_kernel_scalar_t, mul_scalar)

ORT_MATH_RESULT_BINARY_IMPL(multiply,
    ORT_MATH_FRONTEND_DISPATCH_SYMBOL(mul),
    &ort_math_promotion_schema_mul)
ORT_MATH_RESULT_SCALAR_IMPL(multiply,
    ORT_MATH_FRONTEND_DISPATCH_SYMBOL(mul_scalar),
    &ort_math_promotion_schema_mul)