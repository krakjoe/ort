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
@extract %python% %extract.py% -n sub -f subtract -b -w }}} */

static const ONNXTensorElementDataType ort_math_promotion_schema_table_sub[11*11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float16, float16) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float16, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float16, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float16, int8) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float16, int16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float16, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float16, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float16, uint8) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float16, uint16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float16, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float16, bool) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float32, float16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float32, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float32, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float32, int8) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float32, int16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float32, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float32, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float32, uint8) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float32, uint16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float32, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(float32, bool) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float64, float16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float64, float32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float64, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float64, int8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float64, int16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float64, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float64, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float64, uint8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float64, uint16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float64, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(float64, bool) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(int8, float16) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(int8, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(int8, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,	// sub(int8, int8) -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// sub(int8, int16) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(int8, int32) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int8, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// sub(int8, uint8) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(int8, uint16) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int8, uint32) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,	// sub(int8, bool) -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(int16, float16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(int16, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(int16, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// sub(int16, int8) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// sub(int16, int16) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(int16, int32) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int16, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// sub(int16, uint8) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(int16, uint16) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int16, uint32) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// sub(int16, bool) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(int32, float16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(int32, float32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(int32, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(int32, int8) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(int32, int16) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(int32, int32) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int32, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(int32, uint8) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(int32, uint16) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int32, uint32) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(int32, bool) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(int64, float16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(int64, float32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(int64, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int64, int8) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int64, int16) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int64, int32) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int64, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int64, uint8) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int64, uint16) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int64, uint32) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(int64, bool) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(uint8, float16) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(uint8, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(uint8, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// sub(uint8, int8) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// sub(uint8, int16) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(uint8, int32) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(uint8, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,	// sub(uint8, uint8) -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,	// sub(uint8, uint16) -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// sub(uint8, uint32) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,	// sub(uint8, bool) -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(uint16, float16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(uint16, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(uint16, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(uint16, int8) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(uint16, int16) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(uint16, int32) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(uint16, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,	// sub(uint16, uint8) -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,	// sub(uint16, uint16) -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// sub(uint16, uint32) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,	// sub(uint16, bool) -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(uint32, float16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(uint32, float32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(uint32, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(uint32, int8) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(uint32, int16) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(uint32, int32) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(uint32, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// sub(uint32, uint8) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// sub(uint32, uint16) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// sub(uint32, uint32) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// sub(uint32, bool) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(bool, float16) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,	// sub(bool, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,	// sub(bool, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,	// sub(bool, int8) -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// sub(bool, int16) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// sub(bool, int32) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// sub(bool, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,	// sub(bool, uint8) -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,	// sub(bool, uint16) -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// sub(bool, uint32) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,	// sub(bool, bool) -> UNDEFINED
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

static const ort_math_promotion_schema_t ort_math_promotion_schema_sub = {
    .kind     = ORT_MATH_PROMOTION_SCHEMA_KIND_BINARY,
    .operands = ORT_MATH_PROMOTION_SCHEMA_OPERANDS_PROMOTE,
    .table    = ort_math_promotion_schema_table_sub,
    .indices  = ort_math_promotion_schema_indices_sub,
    .size     = 11
};

/* {{{
    !!THIS FILE IS AUTOMATICALLY GENERATED: DO NOT EDIT!!
}}} */
#endif