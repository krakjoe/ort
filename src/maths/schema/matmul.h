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
#ifndef HAVE_ORT_MATHS_SCHEMA_MATMUL
#define HAVE_ORT_MATHS_SCHEMA_MATMUL

#include "maths/promotion.h"

/* {{{ 
@extract python3 tests/fixtures/extract.py -f matmul -n matmul -b
/* }}} */
static const ONNXTensorElementDataType ort_math_promotion_schema_table_matmul[11*11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 matmul float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 matmul float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 matmul float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 matmul int8 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 matmul int16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 matmul int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 matmul int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 matmul uint8 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 matmul uint16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 matmul uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 matmul bool -> float16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 matmul float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 matmul float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 matmul float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 matmul int8 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 matmul int16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 matmul int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 matmul int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 matmul uint8 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 matmul uint16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 matmul uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 matmul bool -> float32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 matmul float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 matmul float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 matmul float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 matmul int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 matmul int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 matmul int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 matmul int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 matmul uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 matmul uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 matmul uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 matmul bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int8 matmul float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int8 matmul float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 matmul float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // int8 matmul int8 -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int8 matmul int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int8 matmul int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int8 matmul int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int8 matmul uint8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int8 matmul uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int8 matmul uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // int8 matmul bool -> int8

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int16 matmul float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int16 matmul float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 matmul float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 matmul int8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 matmul int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int16 matmul int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int16 matmul int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 matmul uint8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int16 matmul uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int16 matmul uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 matmul bool -> int16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 matmul float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 matmul float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 matmul float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 matmul int8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 matmul int16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 matmul int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int32 matmul int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 matmul uint8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 matmul uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int32 matmul uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 matmul bool -> int32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 matmul float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 matmul float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 matmul float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 matmul int8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 matmul int16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 matmul int32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 matmul int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 matmul uint8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 matmul uint16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 matmul uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 matmul bool -> int64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint8 matmul float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint8 matmul float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 matmul float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // uint8 matmul int8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // uint8 matmul int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint8 matmul int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint8 matmul int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // uint8 matmul uint8 -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint8 matmul uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint8 matmul uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // uint8 matmul bool -> uint8

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint16 matmul float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint16 matmul float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 matmul float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 matmul int8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 matmul int16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 matmul int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint16 matmul int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 matmul uint8 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 matmul uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint16 matmul uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 matmul bool -> uint16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 matmul float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 matmul float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 matmul float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 matmul int8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 matmul int16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 matmul int32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 matmul int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 matmul uint8 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 matmul uint16 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 matmul uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 matmul bool -> uint32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // bool matmul float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // bool matmul float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool matmul float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // bool matmul int8 -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // bool matmul int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // bool matmul int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // bool matmul int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // bool matmul uint8 -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // bool matmul uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // bool matmul uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,         // bool matmul bool -> bool

};

static const ONNXTensorElementDataType ort_math_promotion_schema_indices_matmul[11] = {
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

static const ort_math_type_promotion_schema_t ort_math_promotion_schema_matmul = {
    .kind    = ORT_MATH_TYPE_PROMOTION_SCHEMA_BINARY,
    .table   = ort_math_promotion_schema_table_matmul,
    .indices = ort_math_promotion_schema_indices_matmul,
    .size    = 11
};

#endif