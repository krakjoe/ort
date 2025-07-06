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

 #include <math.h>

 #include "status.h"

#include "maths/codegen.h"
#include "maths/core.h"

ort_math_result_t* ort_math_result_sum(ort_tensor_t* tensor, zval* axis_zval, zend_bool keepdims) {
    if (!ort_math_validate_input(tensor, "sum")) {
        return NULL;
    }

    size_t element_count = tensor->elements;
    if (element_count == 0) {
        php_ort_status_throw(php_ort_status_math_error_ce, "sum: cannot sum empty tensor");
        return NULL;
    }

    // If no axis specified, sum all elements
    if (axis_zval == NULL || Z_TYPE_P(axis_zval) == IS_NULL) {
        ort_tensor_t* result_tensor = ort_tensor_create_result(NULL, 0, tensor->type, "sum_result");
        if (!result_tensor) {
            return NULL;
        }

        switch (tensor->type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                float sum = 0.0f;
                float* data = (float*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(float*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
                double sum = 0.0;
                double* data = (double*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(double*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
                int8_t sum = 0;
                int8_t* data = (int8_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(int8_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
                int16_t sum = 0;
                int16_t* data = (int16_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(int16_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                int32_t sum = 0;
                int32_t* data = (int32_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(int32_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                int64_t sum = 0;
                int64_t* data = (int64_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(int64_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
                uint8_t sum = 0;
                uint8_t* data = (uint8_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(uint8_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
                uint16_t sum = 0;
                uint16_t* data = (uint16_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(uint16_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
                uint32_t sum = 0;
                uint32_t* data = (uint32_t*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum += data[i];
                }
                *(uint32_t*)result_tensor->data = sum;
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
                zend_bool sum = 0;
                zend_bool* data = (zend_bool*)tensor->data;
                for (size_t i = 0; i < element_count; i++) {
                    sum = sum || data[i]; // Logical OR for boolean sum
                }
                *(zend_bool*)result_tensor->data = sum;
                break;
            }
            default:
                php_ort_status_throw(php_ort_status_math_invalidtype_ce,
                    "sum: unsupported data type for reduction operation");
                return NULL;
        }

        return ort_math_result_create(result_tensor);
    }

    // Handle specified axis
    if (Z_TYPE_P(axis_zval) != IS_LONG) {
        php_ort_status_throw(php_ort_status_tensor_invalidshape_ce, 
            "sum: axis must be an integer");
        return NULL;
    }

    int64_t axis = Z_LVAL_P(axis_zval);
    if (!ort_math_validate_axis(tensor, axis, "sum")) {
        return NULL;
    }

    // Normalize negative axis
    if (axis < 0) {
        axis += tensor->dimensions;
    }

    // Calculate result shape
    int64_t* result_shape = ecalloc(tensor->dimensions - (keepdims ? 0 : 1), sizeof(int64_t));
    size_t result_dims = 0;
    
    for (size_t i = 0; i < tensor->dimensions; i++) {
        if (i != (size_t)axis) {
            result_shape[result_dims++] = tensor->shape[i];
        } else if (keepdims) {
            result_shape[result_dims++] = 1;
        }
    }
    
    if (result_dims == 0 && !keepdims) {
        result_dims = 1;
        result_shape[0] = 1;
    }

    // Create result tensor
    ort_tensor_t* result_tensor = ort_tensor_create_result(result_shape, result_dims, tensor->type, "sum_result");
    if (!result_tensor) {
        efree(result_shape);
        return NULL;
    }

    // Calculate strides
    size_t* strides = ecalloc(tensor->dimensions, sizeof(size_t));
    strides[tensor->dimensions - 1] = 1;
    for (int64_t i = tensor->dimensions - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * tensor->shape[i + 1];
    }

    // Sum along the specified axis
    size_t axis_size = tensor->shape[axis];
    size_t outer_size = 1;
    for (size_t i = 0; i < (size_t)axis; i++) {
        outer_size *= tensor->shape[i];
    }
    size_t inner_size = strides[axis];

    switch (tensor->type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            float* input = (float*)tensor->data;
            float* output = (float*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            double* input = (double*)tensor->data;
            double* output = (double*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    double sum = 0.0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
            int8_t* input = (int8_t*)tensor->data;
            int8_t* output = (int8_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    int8_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
            int16_t* input = (int16_t*)tensor->data;
            int16_t* output = (int16_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    int16_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            int32_t* input = (int32_t*)tensor->data;
            int32_t* output = (int32_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    int32_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            int64_t* input = (int64_t*)tensor->data;
            int64_t* output = (int64_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    int64_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
            uint8_t* input = (uint8_t*)tensor->data;
            uint8_t* output = (uint8_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    uint8_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
            uint16_t* input = (uint16_t*)tensor->data;
            uint16_t* output = (uint16_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    uint16_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
            uint32_t* input = (uint32_t*)tensor->data;
            uint32_t* output = (uint32_t*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    uint32_t sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum += input[idx];
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            zend_bool* input = (zend_bool*)tensor->data;
            zend_bool* output = (zend_bool*)result_tensor->data;
            for (size_t i = 0; i < outer_size; i++) {
                for (size_t k = 0; k < inner_size; k++) {
                    zend_bool sum = 0;
                    for (size_t j = 0; j < axis_size; j++) {
                        size_t idx = i * (axis_size * inner_size) + j * inner_size + k;
                        sum = sum || input[idx]; // Logical OR for boolean sum
                    }
                    size_t out_idx = i * inner_size + k;
                    output[out_idx] = sum;
                }
            }
            break;
        }
        default:
            php_ort_status_throw(php_ort_status_math_invalidtype_ce,
                "sum: unsupported data type for reduction operation");
            efree(strides);
            efree(result_shape);
            return NULL;
    }

    efree(strides);
    efree(result_shape);

    return ort_math_result_create(result_tensor);
}