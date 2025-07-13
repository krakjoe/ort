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
 @brief Implements frontend dot operations for tensors
 @test tests/math/dot
*/

#include <math.h>

#include "status.h"
#include "alloc.h"

#include "maths/cast.h"
#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/result.h"
#include "maths/schema/dot.h"

#define ORT_MATH_DOT_IMPL_FOR_TYPE(c_type, unused) \
    static zend_always_inline c_type ort_math_dot_impl_##c_type( \
        const c_type* va, const c_type* vb, size_t count) { \
        c_type sum = 0; \
        for (size_t i = 0; i < count; i++) { \
            sum += va[i] * vb[i]; \
        } \
        return sum; \
    }

ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_DOT_IMPL_FOR_TYPE)

#define ORT_MATH_DOT_IMPL(c_type, onnx_type)        \
    ORT_MATH_FRONTEND_BINARY_OP_DECL(dot, c_type) { \
        c_type* res = (c_type*)result;              \
        const c_type* va = (const c_type*)a;        \
        const c_type* vb = (const c_type*)b;        \
        res[0] = ort_math_dot_impl_##c_type(va, vb, count); \
    }

ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_DOT_IMPL)

static ort_math_element_op_func_t ort_math_frontend_get_dot_func(ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return dispatch->dot_func;
}

ort_tensor_t* ort_math_result_dot(ort_tensor_t* a, ort_tensor_t* b) {
    if (!ort_math_validate_input(a, "dot") || !ort_math_validate_input(b, "dot")) {
        return NULL;
    }

    if (a->dimensions != 1 || b->dimensions != 1) {
        php_ort_status_throw(php_ort_status_math_invalidshape_ce,
            "dot: both inputs must be 1D tensors");
        return NULL;
    }

    if (a->elements != b->elements) {
        php_ort_status_throw(php_ort_status_math_invalidshape_ce,
            "dot: vectors must have the same length");
        return NULL;
    }

    ort_math_type_promotion_t promotion = ort_math_type_promote_schema_binary(&ort_math_promotion_schema_dot, a, b);
    if (!promotion.is_valid) {
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,
            "dot: incompatible types for operation");
        return NULL;
    }

    ort_math_element_op_func_t operation = ort_math_frontend_get_dot_func(promotion.result_type);
    if (!operation) {
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,
            "dot: unsupported data type for dot product");
        return NULL;
    }

    int64_t shape[1] = {1};
    ort_tensor_t* result = ort_math_result_tensor(
        shape, 1, promotion.result_type, "dot");

    // Cast input data to promoted type if needed
    size_t element_size = php_ort_tensor_sizeof(result);
    void* a_buf = NULL;
    void* b_buf = NULL;
    const void* a_data = a->data;
    const void* b_data = b->data;
    if (a->type != promotion.result_type) {
        a_buf = ort_alloc(element_size, a->elements);
        for (size_t i = 0; i < a->elements; i++) {
            ort_math_cast_element((char*)a->data + i * php_ort_tensor_sizeof(a),
                                 (char*)a_buf + i * element_size,
                                 a->type, promotion.result_type);
        }
        a_data = a_buf;
    }
    if (b->type != promotion.result_type) {
        b_buf = ort_alloc(element_size, b->elements);
        for (size_t i = 0; i < b->elements; i++) {
            ort_math_cast_element((char*)b->data + i * php_ort_tensor_sizeof(b),
                                 (char*)b_buf + i * element_size,
                                 b->type, promotion.result_type);
        }
        b_data = b_buf;
    }

    operation(result->data, a_data, b_data, a->elements);

    if (a_buf) {
        ort_free(a_buf);
    }

    if (b_buf) {
        ort_free(b_buf);
    }

    return result;
}