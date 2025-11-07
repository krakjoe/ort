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

/**
 * @file codegen/common.h
 * @brief Common code generation macros
 */

#ifndef HAVE_ORT_MATHS_CODEGEN_COMMON
#define HAVE_ORT_MATHS_CODEGEN_COMMON

#define ORT_MATH_FOREACH_SIGNED_TYPE(macro)               \
    macro(int8_t,   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)   \
    macro(int16_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)  \
    macro(int32_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)  \
    macro(int64_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)

#define ORT_MATH_FOREACH_SIGNED_TYPE_EX(macro, ex)            \
    macro(int8_t,   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, ex)   \
    macro(int16_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, ex)  \
    macro(int32_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, ex)  \
    macro(int64_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ex)

#define ORT_MATH_FOREACH_UNSIGNED_TYPE(macro)             \
    macro(uint8_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)  \
    macro(uint16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) \
    macro(uint32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)

#define ORT_MATH_FOREACH_UNSIGNED_TYPE_EX(macro, ex)          \
    macro(uint8_t,  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, ex)  \
    macro(uint16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, ex) \
    macro(uint32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, ex)

#define ORT_MATH_FOREACH_INTEGER_TYPE(macro)              \
    ORT_MATH_FOREACH_SIGNED_TYPE(macro)                   \
    ORT_MATH_FOREACH_UNSIGNED_TYPE(macro)

#define ORT_MATH_FOREACH_INTEGER_TYPE_EX(macro, ex)       \
    ORT_MATH_FOREACH_SIGNED_TYPE_EX(macro, ex)            \
    ORT_MATH_FOREACH_UNSIGNED_TYPE_EX(macro, ex)

#define ORT_MATH_FOREACH_REAL_TYPE(macro)                 \
    macro(float32, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32) \
    macro(float64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64)

#define ORT_MATH_FOREACH_CUSTOM_TYPE(macro)                   \
    macro(float16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)

#define ORT_MATH_FOREACH_REAL_TYPE_EX(macro, ex)              \
    macro(float32, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32, ex) \
    macro(float64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64, ex)

#define ORT_MATH_FOREACH_CUSTOM_TYPE_EX(macro, ex)            \
    macro(float16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, ex) \

#define ORT_MATH_FOREACH_NUMERIC_TYPE(macro)              \
    ORT_MATH_FOREACH_REAL_TYPE(macro)                     \
    ORT_MATH_FOREACH_INTEGER_TYPE(macro)

#define ORT_MATH_FOREACH_NUMERIC_TYPE_EX(macro, ex)       \
    ORT_MATH_FOREACH_REAL_TYPE_EX(macro, ex)              \
    ORT_MATH_FOREACH_INTEGER_TYPE_EX(macro, ex)

#define ORT_MATH_FOREACH_ALL_TYPES(macro)                 \
    ORT_MATH_FOREACH_NUMERIC_TYPE(macro)                  \
    macro(zend_bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) 

#define ORT_MATH_FOREACH_ALL_TYPES_EX(macro, ex)          \
    ORT_MATH_FOREACH_NUMERIC_TYPE_EX(macro, ex)           \
    macro(zend_bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, ex)

#endif