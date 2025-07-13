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
#ifndef HAVE_ORT_MATHS_SCHEMA_MUL
#define HAVE_ORT_MATHS_SCHEMA_MUL

#include "maths/promotion.h"

/* {{{ 
@extract python3 tests/fixtures/extract.py -n mul -f multiply -b
/* }}} */
static const ONNXTensorElementDataType ort_math_promotion_schema_table_mul[11*11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 multiply float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 multiply float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 multiply float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 multiply int8 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 multiply int16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 multiply int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 multiply int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 multiply uint8 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 multiply uint16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 multiply uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 multiply bool -> float16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 multiply float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 multiply float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 multiply float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 multiply int8 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 multiply int16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 multiply int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 multiply int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 multiply uint8 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 multiply uint16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 multiply uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 multiply bool -> float32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 multiply float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 multiply float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 multiply float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 multiply int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 multiply int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 multiply int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 multiply int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 multiply uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 multiply uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 multiply uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 multiply bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int8 multiply float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int8 multiply float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 multiply float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, // int8 multiply int8 -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int8 multiply int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int8 multiply int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int8 multiply int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int8 multiply uint8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int8 multiply uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int8 multiply uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, // int8 multiply bool -> int8

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int16 multiply float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int16 multiply float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 multiply float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 multiply int8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 multiply int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int16 multiply int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int16 multiply int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 multiply uint8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int16 multiply uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int16 multiply uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 multiply bool -> int16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 multiply float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 multiply float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 multiply float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 multiply int8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 multiply int16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 multiply int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int32 multiply int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 multiply uint8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 multiply uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int32 multiply uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 multiply bool -> int32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 multiply float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 multiply float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 multiply float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 multiply int8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 multiply int16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 multiply int32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 multiply int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 multiply uint8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 multiply uint16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 multiply uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 multiply bool -> int64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint8 multiply float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint8 multiply float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 multiply float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // uint8 multiply int8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // uint8 multiply int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint8 multiply int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint8 multiply int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // uint8 multiply uint8 -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint8 multiply uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint8 multiply uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // uint8 multiply bool -> uint8

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint16 multiply float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint16 multiply float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 multiply float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 multiply int8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 multiply int16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 multiply int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint16 multiply int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 multiply uint8 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 multiply uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint16 multiply uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 multiply bool -> uint16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 multiply float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 multiply float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 multiply float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 multiply int8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 multiply int16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 multiply int32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 multiply int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 multiply uint8 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 multiply uint16 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 multiply uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 multiply bool -> uint32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // bool multiply float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // bool multiply float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool multiply float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, // bool multiply int8 -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // bool multiply int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // bool multiply int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // bool multiply int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // bool multiply uint8 -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // bool multiply uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // bool multiply uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, // bool multiply bool -> bool

};

static const ONNXTensorElementDataType ort_math_promotion_schema_indices_mul[11] = {
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

static const ort_math_type_promotion_schema_t ort_math_promotion_schema_mul = {
    .kind    = ORT_MATH_TYPE_PROMOTION_SCHEMA_BINARY,
    .table   = ort_math_promotion_schema_table_mul,
    .indices = ort_math_promotion_schema_indices_mul,
    .size    = 11
};

#endif