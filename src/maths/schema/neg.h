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
#ifndef HAVE_ORT_MATHS_SCHEMA_NEG
#define HAVE_ORT_MATHS_SCHEMA_NEG

#include "maths/promotion.h"

/* {{{ 
@extract python3 tests/fixtures/extract.py -f negative -n neg -u
/* }}} */
static const ONNXTensorElementDataType ort_math_promotion_schema_table_neg[11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // int8 -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // uint8 -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,    // bool -> UNDEFINED
};

static const ONNXTensorElementDataType ort_math_promotion_schema_indices_neg[11] = {
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

static const ort_math_type_promotion_schema_t ort_math_promotion_schema_neg = {
    .table   = ort_math_promotion_schema_table_neg,
    .indices = ort_math_promotion_schema_indices_neg,
    .size    = 11
};

#endif