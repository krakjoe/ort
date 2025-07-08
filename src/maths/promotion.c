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

#include "maths/promotion.h"

/* Helper function to check if a type is a signed integer */
static zend_always_inline int ort_math_promote_is_signed(ONNXTensorElementDataType type) {
    return (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
            type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 ||
            type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
            type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
}

/* Helper function to check if a type is an unsigned integer */
static zend_always_inline int ort_math_promote_is_unsigned(ONNXTensorElementDataType type) {
    return (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
            type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ||
            type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32);
}

/* Helper function to get the larger signed integer type */
static zend_always_inline ONNXTensorElementDataType ort_math_promote_signed(ONNXTensorElementDataType type_a, ONNXTensorElementDataType type_b) {
    /* Promotion order: INT8 < INT16 < INT32 < INT64 */
    if (type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 || type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    if (type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 || type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    }
    if (type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 || type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
}

/* Helper function to get the larger unsigned integer type */
static zend_always_inline ONNXTensorElementDataType ort_math_promote_unsigned(ONNXTensorElementDataType type_a, ONNXTensorElementDataType type_b) {
    /* Promotion order: UINT8 < UINT16 < UINT32 */
    if (type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 || type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    }
    if (type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 || type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
}


/* Type promotion implementation (permissive, for elementwise ops) */
ort_math_type_promotion_t ort_math_type_promote(
    ONNXTensorElementDataType type_a,
    ONNXTensorElementDataType type_b
) {
    ort_math_type_promotion_t result = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, 0};

    /* Handle bool special cases first */
    if (type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        result.result_type = type_b;
        result.is_valid = 1;
        return result;
    }
    if (type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        result.result_type = type_a;
        result.is_valid = 1;
        return result;
    }
    /* Same types */
    if (type_a == type_b) {
        result.result_type = type_a;
        result.is_valid = 1;
        return result;
    }
    /* Type promotion hierarchy (simplified NumPy-style) */
    /* BOOL < INT8 < INT16 < INT32 < INT64 */
    /* UINT8 < UINT16 < UINT32 */  
    /* FLOAT < DOUBLE */
    /* Handle floating point types first */
    if (type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE || type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        result.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        result.is_valid = 1;
    } else if (type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        result.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        result.is_valid = 1;
    }
    /* Handle signed integer types */
    else if (ort_math_promote_is_signed(type_a) &&
             ort_math_promote_is_signed(type_b)) {
        result.result_type = ort_math_promote_signed(type_a, type_b);
        result.is_valid = 1;
    }
    /* Handle unsigned integer types */
    else if (ort_math_promote_is_unsigned(type_a) &&
             ort_math_promote_is_unsigned(type_b)) {
        result.result_type = ort_math_promote_unsigned(type_a, type_b);
        result.is_valid = 1;
    }
    /* Mixed signed/unsigned - promote to INT64 for safety */
    else if ((ort_math_promote_is_signed(type_a)    &&
              ort_math_promote_is_unsigned(type_b)) ||
             (ort_math_promote_is_unsigned(type_a)  &&
              ort_math_promote_is_signed(type_b))) {
        result.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        result.is_valid = 1;
    }
    return result;
}

/* Strict type promotion for dot/matmul/reductions (NumPy/ONNX style) */
ort_math_type_promotion_t ort_math_type_promote_strict(
    ONNXTensorElementDataType type_a,
    ONNXTensorElementDataType type_b
) {
    ort_math_type_promotion_t result = {
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, 0};

    /* Handle bool special cases first */
    if (type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        result.result_type = type_b;
        result.is_valid = 1;
        return result;
    }
    if (type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        result.result_type = type_a;
        result.is_valid = 1;
        return result;
    }

    /* Same types */
    if (type_a == type_b) {
        result.result_type = type_a;
        result.is_valid = 1;
        return result;
    }

    /* Type promotion hierarchy (simplified NumPy-style) */
    /* BOOL < INT8 < INT16 < INT32 < INT64 */
    /* UINT8 < UINT16 < UINT32 */  
    /* FLOAT < DOUBLE */

    /* Handle floating point types first */
    if (type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
        type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        result.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        result.is_valid = 1;
    } else if (type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        result.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        result.is_valid = 1;
    }

    /* Handle signed integer types */
    else if (ort_math_promote_is_signed(type_a) &&
             ort_math_promote_is_signed(type_b)) {
        result.result_type = ort_math_promote_signed(type_a, type_b);
        result.is_valid = 1;
    }
    /* Handle unsigned integer types */
    else if (ort_math_promote_is_unsigned(type_a) &&
             ort_math_promote_is_unsigned(type_b)) {
        result.result_type = ort_math_promote_unsigned(type_a, type_b);
        result.is_valid = 1;
    }
    /* Mixed signed/unsigned - follow ONNX/NumPy promotion rules */
    else if ((ort_math_promote_is_signed(type_a)    &&
              ort_math_promote_is_unsigned(type_b)) ||
             (ort_math_promote_is_unsigned(type_a)  && 
              ort_math_promote_is_signed(type_b))) {
        // int8 + uint8 => int32
        if ((type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 && type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) ||
            (type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 && type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)) {
            result.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            result.is_valid = 1;
        }
        // int16 + uint16 => int32
        else if ((type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 && type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) ||
                 (type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 && type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
            result.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            result.is_valid = 1;
        }
        // int32 + uint32 => int64
        else if ((type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) ||
                 (type_b == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && type_a == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)) {
            result.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            result.is_valid = 1;
        }
        // All other mixed signed/unsigned cases are not supported (e.g., int64 + uint32, etc.)
        else {
            result.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            result.is_valid = 0;
        }
    }

    return result;
}
