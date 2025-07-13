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
#ifndef HAVE_ORT_MATHS_SCHEMA_SUB
#define HAVE_ORT_MATHS_SCHEMA_SUB

#include "maths/promotion.h"

/* {{{ 
@extract python3 tests/fixtures/extract.py -n sub -f subtract -b
/* }}} */
static const ONNXTensorElementDataType ort_math_promotion_schema_table_sub[11*11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 subtract float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 subtract float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 subtract float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 subtract int8 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 subtract int16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 subtract int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 subtract int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 subtract uint8 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 subtract uint16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 subtract uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 subtract bool -> float16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 subtract float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 subtract float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 subtract float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 subtract int8 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 subtract int16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 subtract int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 subtract int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 subtract uint8 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 subtract uint16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 subtract uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 subtract bool -> float32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 subtract float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 subtract float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 subtract float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 subtract int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 subtract int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 subtract int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 subtract int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 subtract uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 subtract uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 subtract uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 subtract bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int8 subtract float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int8 subtract float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 subtract float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // int8 subtract int8 -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int8 subtract int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int8 subtract int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int8 subtract int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int8 subtract uint8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int8 subtract uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int8 subtract uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // int8 subtract bool -> int8

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int16 subtract float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int16 subtract float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 subtract float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 subtract int8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 subtract int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int16 subtract int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int16 subtract int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 subtract uint8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int16 subtract uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int16 subtract uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 subtract bool -> int16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 subtract float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 subtract float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 subtract float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 subtract int8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 subtract int16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 subtract int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int32 subtract int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 subtract uint8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 subtract uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int32 subtract uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 subtract bool -> int32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 subtract float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 subtract float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 subtract float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 subtract int8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 subtract int16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 subtract int32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 subtract int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 subtract uint8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 subtract uint16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 subtract uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 subtract bool -> int64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint8 subtract float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint8 subtract float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 subtract float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // uint8 subtract int8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // uint8 subtract int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint8 subtract int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint8 subtract int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // uint8 subtract uint8 -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint8 subtract uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint8 subtract uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // uint8 subtract bool -> uint8

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint16 subtract float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint16 subtract float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 subtract float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 subtract int8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 subtract int16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 subtract int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint16 subtract int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 subtract uint8 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 subtract uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint16 subtract uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 subtract bool -> uint16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 subtract float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 subtract float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 subtract float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 subtract int8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 subtract int16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 subtract int32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 subtract int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 subtract uint8 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 subtract uint16 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 subtract uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 subtract bool -> uint32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // bool subtract float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // bool subtract float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool subtract float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // bool subtract int8 -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // bool subtract int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // bool subtract int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // bool subtract int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // bool subtract uint8 -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // bool subtract uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // bool subtract uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,    // bool subtract bool -> UNDEFINED

};

static const ONNXTensorElementDataType ort_math_promotion_schema_indices_sub[11] = {
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

static const ort_math_type_promotion_schema_t ort_math_promotion_schema_sub = {
    .kind    = ORT_MATH_TYPE_PROMOTION_SCHEMA_BINARY,
    .table   = ort_math_promotion_schema_table_sub,
    .indices = ort_math_promotion_schema_indices_sub,
    .size    = 11
};

#endif