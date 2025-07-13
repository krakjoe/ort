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
#ifndef HAVE_ORT_MATHS_SCHEMA_DIV
#define HAVE_ORT_MATHS_SCHEMA_DIV

#include "maths/promotion.h"

/* {{{ 
@extract python3 tests/fixtures/extract.py -f divide -n div -b
/* }}} */
static const ONNXTensorElementDataType ort_math_promotion_schema_table_div[11*11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 divide float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 divide float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 divide float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 divide int8 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 divide int16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 divide int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 divide int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 divide uint8 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 divide uint16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float16 divide uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float16 divide bool -> float16

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 divide float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 divide float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 divide float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 divide int8 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 divide int16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 divide int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 divide int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 divide uint8 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 divide uint16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float32 divide uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // float32 divide bool -> float32

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 divide float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 divide float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 divide float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 divide int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 divide int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 divide int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 divide int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 divide uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 divide uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 divide uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // float64 divide bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int8 divide float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int8 divide float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 divide float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 divide int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 divide int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 divide int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 divide int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 divide uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 divide uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 divide uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int8 divide bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int16 divide float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // int16 divide float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 divide float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 divide int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 divide int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 divide int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 divide int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 divide uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 divide uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 divide uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int16 divide bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 divide float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 divide float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 divide float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 divide int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 divide int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 divide int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 divide int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 divide uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 divide uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 divide uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int32 divide bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 divide float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 divide float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 divide float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 divide int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 divide int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 divide int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 divide int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 divide uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 divide uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 divide uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // int64 divide bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint8 divide float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint8 divide float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 divide float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 divide int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 divide int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 divide int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 divide int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 divide uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 divide uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 divide uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint8 divide bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint16 divide float16 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // uint16 divide float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 divide float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 divide int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 divide int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 divide int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 divide int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 divide uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 divide uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 divide uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint16 divide bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 divide float16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 divide float32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 divide float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 divide int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 divide int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 divide int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 divide int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 divide uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 divide uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 divide uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // uint32 divide bool -> float64

    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // bool divide float16 -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,        // bool divide float32 -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool divide float64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool divide int8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool divide int16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool divide int32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool divide int64 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool divide uint8 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool divide uint16 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool divide uint32 -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,       // bool divide bool -> float64

};

static const ONNXTensorElementDataType ort_math_promotion_schema_indices_div[11] = {
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

static const ort_math_type_promotion_schema_t ort_math_promotion_schema_div = {
    .kind    = ORT_MATH_TYPE_PROMOTION_SCHEMA_BINARY,
    .table   = ort_math_promotion_schema_table_div,
    .indices = ort_math_promotion_schema_indices_div,
    .size    = 11
};

#endif