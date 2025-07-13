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
#ifndef HAVE_ORT_MATHS_SCHEMA_MOD
#define HAVE_ORT_MATHS_SCHEMA_MOD

#include "maths/promotion.h"

/* {{{ 
@extract python3 tests/fixtures/extract.py -f mod -n mod -b
/* }}} */
static const ONNXTensorElementDataType ort_math_promotion_schema_table_mod[11*11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 mod float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 mod float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 mod float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 mod int8 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 mod int16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 mod int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 mod int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 mod uint8 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 mod uint16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 mod uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 mod bool -> float16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 mod float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 mod float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 mod float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 mod int8 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 mod int16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 mod int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 mod int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 mod uint8 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 mod uint16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 mod uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 mod bool -> float32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 mod float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 mod float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 mod float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 mod int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 mod int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 mod int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 mod int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 mod uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 mod uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 mod uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 mod bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int8 mod float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int8 mod float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 mod float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // int8 mod int8 -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int8 mod int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int8 mod int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int8 mod int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int8 mod uint8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int8 mod uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int8 mod uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // int8 mod bool -> int8

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int16 mod float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int16 mod float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 mod float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 mod int8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 mod int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int16 mod int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int16 mod int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 mod uint8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int16 mod uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int16 mod uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // int16 mod bool -> int16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 mod float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 mod float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 mod float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 mod int8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 mod int16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 mod int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int32 mod int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 mod uint8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 mod uint16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int32 mod uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // int32 mod bool -> int32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 mod float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 mod float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 mod float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 mod int8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 mod int16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 mod int32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 mod int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 mod uint8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 mod uint16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 mod uint32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // int64 mod bool -> int64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint8 mod float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint8 mod float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 mod float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // uint8 mod int8 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // uint8 mod int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint8 mod int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint8 mod int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // uint8 mod uint8 -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint8 mod uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint8 mod uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // uint8 mod bool -> uint8

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint16 mod float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint16 mod float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 mod float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 mod int8 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 mod int16 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // uint16 mod int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint16 mod int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 mod uint8 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 mod uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint16 mod uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // uint16 mod bool -> uint16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 mod float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 mod float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 mod float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 mod int8 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 mod int16 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 mod int32 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // uint32 mod int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 mod uint8 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 mod uint16 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 mod uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // uint32 mod bool -> uint32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // bool mod float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // bool mod float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool mod float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // bool mod int8 -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,        // bool mod int16 -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,        // bool mod int32 -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,        // bool mod int64 -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,        // bool mod uint8 -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,       // bool mod uint16 -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,       // bool mod uint32 -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,         // bool mod bool -> int8

};

static const ONNXTensorElementDataType ort_math_promotion_schema_indices_mod[11] = {
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

static const ort_math_type_promotion_schema_t ort_math_promotion_schema_mod = {
    .table   = ort_math_promotion_schema_table_mod,
    .indices = ort_math_promotion_schema_indices_mod,
    .size    = 11
};

#endif