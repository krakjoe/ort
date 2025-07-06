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
#include <math.h>

#include "status.h"

#include "maths/codegen.h"
#include "maths/core.h"

/* =============================================================================
 * SQRT OPERATIONS
 * =============================================================================
 */

#define ORT_MATH_SQRT_IMPL_FOR_TYPE(c_type, unused)                          \
    static zend_always_inline c_type ort_math_sqrt_impl_##c_type(c_type a) { \
        if (a < 0) {                                                         \
            return 0;                                                        \
        }                                                                    \
        if (a == 0 || a == 1) {                                              \
            return a;                                                        \
        }                                                                    \
        c_type x = a;                                                        \
        c_type y = (x + 1) / 2;                                              \
        while (y < x) {                                                      \
            x = y;                                                           \
            y = (x + a / x) / 2;                                             \
        }                                                                    \
        return x;                                                            \
}

ORT_MATH_FOREACH_INTEGER_TYPE(ORT_MATH_SQRT_IMPL_FOR_TYPE)

#define ORT_MATH_SQRT_INTEGER_IMPL(c_type, onnx_type)                \
    void ort_math_ops_sqrt_##c_type(                                 \
        void* result, const void* a, size_t count) {                 \
    c_type* res = (c_type*)result;                                   \
    const c_type* va = (const c_type*)a;                             \
    for (size_t i = 0; i < count; i++) {                             \
        res[i] = ort_math_sqrt_impl_##c_type(                        \
            va[i]);                                                  \
    }                                                                \
}

ORT_MATH_FOREACH_INTEGER_TYPE(ORT_MATH_SQRT_INTEGER_IMPL)

#define ORT_MATH_SQRT_REAL_IMPL(ctype, onnx_type)                           \
void ort_math_ops_sqrt_##ctype(void* result, const void* a, size_t count) { \
    ctype* res = (ctype*)result;                                            \
    const ctype* va = (const ctype*)a;                                      \
    for (size_t i = 0; i < count; i++) {                                    \
        res[i] = sqrt(va[i]);                                               \
    }                                                                       \
}

ORT_MATH_FOREACH_REAL_TYPE(ORT_MATH_SQRT_REAL_IMPL)

static ort_math_unary_op_func_t ort_math_ops_get_sqrt_func(ONNXTensorElementDataType type) {
    const ort_math_type_dispatch_t* dispatch =
        ort_math_get_dispatch(type);
    return dispatch->sqrt_func;
}

ORT_MATH_UNARY_RESULT_IMPL(sqrt, ort_math_ops_get_sqrt_func)