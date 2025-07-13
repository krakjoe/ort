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

#ifndef HAVE_ORT_MATHS_PROMOTION
#define HAVE_ORT_MATHS_PROMOTION

#include "tensor.h"
#include "maths/core.h"

typedef struct _ort_math_type_promotion_t {
    ONNXTensorElementDataType result_type;
    zend_bool is_valid;
    struct {
        ort_tensor_t* inputs[2];
        size_t        count;
    } upcast;
} ort_math_type_promotion_t;

typedef enum _ort_math_type_promotion_schema_kind_t {
    ORT_MATH_TYPE_PROMOTION_SCHEMA_INVALID = 0,
    ORT_MATH_TYPE_PROMOTION_SCHEMA_BINARY  = 1,
    ORT_MATH_TYPE_PROMOTION_SCHEMA_UNARY   = 2
} ort_math_type_promotion_schema_kind_t;

typedef struct _ort_math_type_promotion_schema_t {
    ort_math_type_promotion_schema_kind_t kind;

    // 2D mapping: [a_type][b_type] = result_type
    // Types are ONNXTensorElementDataType, but must be mapped to indices by the frontend unit
    const ONNXTensorElementDataType* table;    // flat array [size * size]
    const ONNXTensorElementDataType* indices;  // array [size]
    int size; // length of indices
} ort_math_type_promotion_schema_t;

ONNXTensorElementDataType ort_math_type_promotion_schema_resolve_binary(
    const ort_math_type_promotion_schema_t* schema,
    ONNXTensorElementDataType type_a,
    ONNXTensorElementDataType type_b
);

ort_math_type_promotion_t ort_math_type_promote_schema_binary(
    const ort_math_type_promotion_schema_t* schema,
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b
);

ONNXTensorElementDataType ort_math_type_promotion_schema_resolve_unary(
    const ort_math_type_promotion_schema_t* schema,
    ONNXTensorElementDataType type_a
);

ort_math_type_promotion_t ort_math_type_promote_schema_unary(
    const ort_math_type_promotion_schema_t* schema,
    ort_tensor_t* tensor
);

ort_math_type_promotion_t ort_math_type_promote_schema_scalar(
    const ort_math_type_promotion_schema_t* schema,
    ort_tensor_t* tensor,
    zval* scalar
);

ort_math_type_promotion_t ort_math_type_promote(
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b
);

ort_math_type_promotion_t ort_math_operation_promote(
    ONNXTensorElementDataType type,
    size_t inputs,
    ... /* ort_tensor_t* inputs[] */
);

void* ort_math_operation_upcast(
    const ort_tensor_t* result,
    const ort_math_type_promotion_t* promotion,
    void* data
);


void* ort_math_operation_broadcast(
    const ort_tensor_t* result,
    const struct _ort_math_type_promotion_t* promotion,
    const ort_tensor_t* input,
    void* out_buf
);

#endif