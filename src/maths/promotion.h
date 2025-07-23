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

typedef struct _ort_math_promotion_t {
    ONNXTensorElementDataType result_type;
    zend_bool is_valid;
    struct {
        ort_tensor_t* inputs[2];
        size_t        count;
    } upcast;
    struct {
        ort_tensor_t* lhs;
        ort_tensor_t* rhs;
    } inputs;
} ort_math_promotion_t;

typedef enum _ort_math_promotion_schema_kind_t {
    ORT_MATH_PROMOTION_SCHEMA_INVALID = 0,
    ORT_MATH_PROMOTION_SCHEMA_KIND_BINARY  = 1,
    ORT_MATH_PROMOTION_SCHEMA_KIND_UNARY   = 2
} ort_math_promotion_schema_kind_t;

typedef enum _ort_math_promotion_schema_operands_t {
    ORT_MATH_PROMOTION_SCHEMA_OPERANDS_PROMOTE,
    ORT_MATH_PROMOTION_SCHEMA_OPERANDS_PRESERVE
} ort_math_promotion_schema_operands_t;

typedef struct _ort_math_promotion_schema_t {
    ort_math_promotion_schema_kind_t kind;
    ort_math_promotion_schema_operands_t operands;

    // 2D mapping: [a_type][b_type] = result_type
    // Types are ONNXTensorElementDataType, but must be mapped to indices by the frontend unit
    const ONNXTensorElementDataType* table;    // flat array [size * size]
    const ONNXTensorElementDataType* indices;  // array [size]
    int size; // length of indices
} ort_math_promotion_schema_t;

ONNXTensorElementDataType ort_math_promotion_resolve_binary(
    const ort_math_promotion_schema_t* schema,
    ONNXTensorElementDataType type_a,
    ONNXTensorElementDataType type_b
);

ONNXTensorElementDataType ort_math_promotion_resolve_unary(
    const ort_math_promotion_schema_t* schema,
    ONNXTensorElementDataType type_a
);

ort_math_promotion_t ort_math_promotion_perform_binary(
    const ort_math_promotion_schema_t* schema,
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b
);

ort_math_promotion_t ort_math_promotion_perform_unary(
    const ort_math_promotion_schema_t* schema,
    ort_tensor_t* tensor
);

ort_math_promotion_t ort_math_promotion_perform_scalar(
    const ort_math_promotion_schema_t* schema,
    ort_tensor_t* tensor,
    zval* scalar
);

void* ort_math_operation_upcast(
    const ort_tensor_t* result,
    const ort_math_promotion_t* promotion,
    void* data
);
#endif