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
#ifndef HAVE_ORT_MATHS_SCHEMA_DOT
#define HAVE_ORT_MATHS_SCHEMA_DOT

#include "maths/promotion.h"

/* {{{
@extract %python% %extract.py% -f multiply -n dot -b -w }}} */

static const ONNXTensorElementDataType ort_math_promotion_schema_table_dot[11*11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,	// dot(float16, float16) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(float16, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float16, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,	// dot(float16, int8) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(float16, int16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float16, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float16, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,	// dot(float16, uint8) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(float16, uint16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float16, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,	// dot(float16, bool) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(float32, float16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(float32, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float32, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(float32, int8) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(float32, int16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float32, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float32, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(float32, uint8) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(float32, uint16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float32, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(float32, bool) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float64, float16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float64, float32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float64, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float64, int8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float64, int16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float64, int32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float64, int64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float64, uint8) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float64, uint16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float64, uint32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(float64, bool) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,	// dot(int8, float16) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(int8, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(int8, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,	// dot(int8, int8) -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// dot(int8, int16) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(int8, int32) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int8, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// dot(int8, uint8) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(int8, uint16) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int8, uint32) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,	// dot(int8, bool) -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(int16, float16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(int16, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(int16, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// dot(int16, int8) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// dot(int16, int16) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(int16, int32) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int16, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// dot(int16, uint8) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(int16, uint16) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int16, uint32) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// dot(int16, bool) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(int32, float16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(int32, float32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(int32, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(int32, int8) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(int32, int16) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(int32, int32) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int32, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(int32, uint8) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(int32, uint16) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int32, uint32) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(int32, bool) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(int64, float16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(int64, float32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(int64, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int64, int8) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int64, int16) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int64, int32) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int64, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int64, uint8) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int64, uint16) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int64, uint32) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(int64, bool) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,	// dot(uint8, float16) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(uint8, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(uint8, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// dot(uint8, int8) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// dot(uint8, int16) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(uint8, int32) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(uint8, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,	// dot(uint8, uint8) -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,	// dot(uint8, uint16) -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// dot(uint8, uint32) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,	// dot(uint8, bool) -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(uint16, float16) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(uint16, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(uint16, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(uint16, int8) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(uint16, int16) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(uint16, int32) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(uint16, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,	// dot(uint16, uint8) -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,	// dot(uint16, uint16) -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// dot(uint16, uint32) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,	// dot(uint16, bool) -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(uint32, float16) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(uint32, float32) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(uint32, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(uint32, int8) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(uint32, int16) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(uint32, int32) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(uint32, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// dot(uint32, uint8) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// dot(uint32, uint16) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// dot(uint32, uint32) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// dot(uint32, bool) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,	// dot(bool, float16) -> float16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,	// dot(bool, float32) -> float32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,	// dot(bool, float64) -> float64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,	// dot(bool, int8) -> int8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,	// dot(bool, int16) -> int16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,	// dot(bool, int32) -> int32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,	// dot(bool, int64) -> int64
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,	// dot(bool, uint8) -> uint8
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,	// dot(bool, uint16) -> uint16
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,	// dot(bool, uint32) -> uint32
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,	// dot(bool, bool) -> bool
};

static const ONNXTensorElementDataType ort_math_promotion_schema_indices_dot[11] = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
};

static const ort_math_promotion_schema_t ort_math_promotion_schema_dot = {
    .kind     = ORT_MATH_PROMOTION_SCHEMA_KIND_BINARY,
    .operands = ORT_MATH_PROMOTION_SCHEMA_OPERANDS_PROMOTE,
    .table    = ort_math_promotion_schema_table_dot,
    .indices  = ort_math_promotion_schema_indices_dot,
    .size     = 11
};

/* {{{
    !!THIS FILE IS AUTOMATICALLY GENERATED: DO NOT EDIT!!
}}} */
#endif