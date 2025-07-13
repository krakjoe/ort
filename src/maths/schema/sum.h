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
#ifndef HAVE_ORT_MATHS_SCHEMA_SUM
#define HAVE_ORT_MATHS_SCHEMA_SUM

#include "maths/promotion.h"

/* {{{ 
@extract python3 tests/fixtures/extract.py -n sum -f sum -u
/* }}} */
static const ONNXTensorElementDataType ort_math_promotion_schema_table_sum[11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,    // uint8 -> uint64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,    // uint16 -> uint64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,    // uint32 -> uint64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // bool -> int64
};

static const ONNXTensorElementDataType ort_math_promotion_schema_indices_sum[11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
};

static const ort_math_promotion_schema_t ort_math_promotion_schema_sum = {
    .kind    = ORT_MATH_TYPE_PROMOTION_SCHEMA_UNARY,
    .table   = ort_math_promotion_schema_table_sum,
    .indices = ort_math_promotion_schema_indices_sum,
    .size    = 11
};

#endif