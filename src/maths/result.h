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

#ifndef HAVE_ORT_MATHS_RESULT
#define HAVE_ORT_MATHS_RESULT

#include "tensor.h"
#include "maths/core.h"

/* Element-wise binary operation */
ort_tensor_t* ort_math_result_element_wise_binary(
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b,
    ort_math_element_op_func_t operation,
    const char* operation_name
);

/* Element-wise scalar operation */
ort_tensor_t* ort_math_result_element_wise_scalar(
    ort_tensor_t* tensor,
    zval* scalar,
    ort_math_scalar_op_func_t operation,
    const char* operation_name
);

/* Element-wise unary operation */
ort_tensor_t* ort_math_result_element_wise_unary(
    ort_tensor_t* tensor,
    ort_math_unary_op_func_t operation,
    const char* operation_name
);

/* Tensor creation for result */
ort_tensor_t* ort_math_result_tensor(
    const int64_t* shape,
    size_t dimensions,
    ONNXTensorElementDataType type,
    const char* name_prefix);

ort_tensor_t** ort_math_result_create(ort_tensor_t* tensor);
void ort_math_result_free(ort_tensor_t** result);

/* Utility functions */
static zend_always_inline size_t ort_math_result_total(const int64_t* shape, size_t dimensions) {
    size_t total = 1;
    for (size_t i = 0; i < dimensions; i++) {
        total *= shape[i];
    }
    return total;
}

static zend_always_inline void ort_math_result_multi(zend_long flat_index, const int64_t* shape, size_t dimensions, int64_t* indices) {
    /* Convert flat index back to multi-dimensional indices */
    for (size_t i = dimensions; i > 0; i--) {
        indices[i-1] = flat_index % shape[i-1];
        flat_index /= shape[i-1];
    }
}

static zend_always_inline zend_long ort_math_result_flat(const int64_t* indices, const int64_t* shape, size_t dimensions) {
    zend_long flat_index = 0;
    zend_long stride = 1;
    
    /* Calculate flat index using row-major order */
    for (size_t i = dimensions; i > 0; i--) {
        flat_index += indices[i-1] * stride;
        stride *= shape[i-1];
    }
    
    return flat_index;
}
#endif