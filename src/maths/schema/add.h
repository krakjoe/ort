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
#ifndef HAVE_ORT_MATHS_SCHEMA_ADD
#define HAVE_ORT_MATHS_SCHEMA_ADD

#include "maths/promotion.h"

/* {{{ 
@extract python3 tests/fixtures/extract.py -f add -n add -b
/* }}} */
static const ONNXTensorElementDataType ort_math_promotion_schema_table_add[11*11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 add float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 add float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 add float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 add int8 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 add int16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 add int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 add int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 add uint8 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 add uint16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 add uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 add bool -> float16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 add float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 add float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 add float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 add int8 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 add int16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 add int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 add int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 add uint8 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 add uint16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 add uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 add bool -> float32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 add float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 add float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 add float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 add int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 add int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 add int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 add int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 add uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 add uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 add uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 add bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int8 add float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int8 add float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 add float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // int8 add int8 -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int8 add int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int8 add int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int8 add int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int8 add uint8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int8 add uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int8 add uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // int8 add bool -> int8

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int16 add float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int16 add float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 add float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 add int8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 add int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int16 add int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int16 add int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 add uint8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int16 add uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int16 add uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 add bool -> int16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 add float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 add float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 add float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 add int8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 add int16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 add int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int32 add int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 add uint8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 add uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int32 add uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 add bool -> int32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 add float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 add float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 add float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 add int8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 add int16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 add int32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 add int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 add uint8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 add uint16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 add uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 add bool -> int64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint8 add float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint8 add float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 add float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // uint8 add int8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // uint8 add int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint8 add int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint8 add int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // uint8 add uint8 -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint8 add uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint8 add uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // uint8 add bool -> uint8

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint16 add float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint16 add float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 add float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 add int8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 add int16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 add int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint16 add int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 add uint8 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 add uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint16 add uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 add bool -> uint16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 add float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 add float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 add float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 add int8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 add int16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 add int32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 add int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 add uint8 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 add uint16 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 add uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 add bool -> uint32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // bool add float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // bool add float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool add float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // bool add int8 -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // bool add int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // bool add int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // bool add int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // bool add uint8 -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // bool add uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // bool add uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,         // bool add bool -> bool

};

static const ONNXTensorElementDataType ort_math_promotion_schema_indices_add[11] = {
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

static const ort_math_type_promotion_schema_t ort_math_promotion_schema_add = {
    .kind    = ORT_MATH_TYPE_PROMOTION_SCHEMA_BINARY,
    .table   = ort_math_promotion_schema_table_add,
    .indices = ort_math_promotion_schema_indices_add,
    .size    = 11
};

#endif