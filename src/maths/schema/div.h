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
@extract %python% %extract.py% -f divide -n div -b -w }}} */

static const ONNXTensorElementDataType ort_math_promotion_schema_table_div[11*11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float16, float16) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float16, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float16, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float16, int8) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float16, int16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float16, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float16, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float16, uint8) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float16, uint16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float16, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float16, bool) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float32, float16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float32, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float32, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float32, int8) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float32, int16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float32, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float32, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float32, uint8) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float32, uint16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float32, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(float32, bool) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float64, float16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float64, float32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float64, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float64, int8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float64, int16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float64, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float64, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float64, uint8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float64, uint16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float64, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(float64, bool) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(int8, float16) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(int8, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int8, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int8, int8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int8, int16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int8, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int8, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int8, uint8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int8, uint16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int8, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int8, bool) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(int16, float16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(int16, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int16, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int16, int8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int16, int16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int16, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int16, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int16, uint8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int16, uint16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int16, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int16, bool) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int32, float16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int32, float32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int32, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int32, int8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int32, int16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int32, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int32, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int32, uint8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int32, uint16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int32, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int32, bool) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int64, float16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int64, float32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int64, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int64, int8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int64, int16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int64, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int64, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int64, uint8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int64, uint16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int64, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(int64, bool) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(uint8, float16) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(uint8, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint8, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint8, int8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint8, int16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint8, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint8, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint8, uint8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint8, uint16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint8, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint8, bool) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(uint16, float16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(uint16, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint16, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint16, int8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint16, int16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint16, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint16, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint16, uint8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint16, uint16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint16, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint16, bool) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint32, float16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint32, float32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint32, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint32, int8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint32, int16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint32, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint32, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint32, uint8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint32, uint16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint32, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(uint32, bool) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(bool, float16) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// div(bool, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(bool, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(bool, int8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(bool, int16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(bool, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(bool, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(bool, uint8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(bool, uint16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(bool, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// div(bool, bool) -> float64
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

static const ort_math_promotion_schema_t ort_math_promotion_schema_div = {
    .kind     = ORT_MATH_PROMOTION_SCHEMA_KIND_BINARY,
    .operands = ORT_MATH_PROMOTION_SCHEMA_OPERANDS_PROMOTE,
    .table    = ort_math_promotion_schema_table_div,
    .indices  = ort_math_promotion_schema_indices_div,
    .size     = 11
};

/* {{{
    !!THIS FILE IS AUTOMATICALLY GENERATED: DO NOT EDIT!!
}}} */
#endif