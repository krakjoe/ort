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

#include "status.h"
#include "maths/broadcast.h"
#include "maths/cast.h"
#include "maths/promotion.h"
#include "maths/result.h"
#include "maths/validate.h"
#include "maths/pool.h"

/* Stack allocation threshold for math optimizations */
#define ORT_MATH_RESULT_STACK_DIMENSIONS 8

static zend_always_inline ort_tensor_t* 
    ort_math_result_element_wise_binary_fast(
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b,
    ort_math_element_op_func_t operation,
    const char* operation_name
) {
    /* Create result tensor with same shape and type as inputs */
    ort_tensor_t* result = ort_math_result_tensor(
        tensor_a->shape, 
        tensor_a->dimensions,
        tensor_a->type, operation_name);

    size_t elements = tensor_a->elements;
    size_t element_size = php_ort_type_sizeof(tensor_a->type);
    size_t pool_size = ort_pool_cores();
    size_t chunk = (elements + pool_size - 1) / pool_size;
    size_t num_chunks = (elements + chunk - 1) / chunk;

    ort_pool_binary_ctx_t ctx = {
        .layout = {
            .element = element_size,
            .total = elements,
            .chunk = chunk
        },
        .result = result->data,
        .a = tensor_a->data,
        .b = tensor_b->data,
        .op = operation
    };

    ort_pool_submit(ort_pool_binary_worker, &ctx, num_chunks);

    return result;
}

/* Element-wise operation helpers implementation */
ort_tensor_t* ort_math_result_element_wise_binary(
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b,
    ort_math_element_op_func_t operation,
    const char* operation_name
) {
    /* Fast path: identical shapes and types */
    if (ort_math_validate_identity(tensor_a, tensor_b)) {
        return ort_math_result_element_wise_binary_fast(
            tensor_a, tensor_b, operation, operation_name);
    }

    /* Fall back to current broadcasting implementation */
    ort_math_broadcast_info_t* binfo = ort_math_broadcast_calculate(
        tensor_a->shape, tensor_a->dimensions,
        tensor_b->shape, tensor_b->dimensions);

    php_ort_status_flow(
        !binfo->is_compatible,
        {
            ort_math_broadcast_free(binfo);
            return NULL;
        },
        php_ort_status_math_invalidshape_ce,
        "%s: tensor shapes are not broadcast compatible",
        operation_name
    );

    /* Type promotion */
    ort_math_type_promotion_t promotion = ort_math_type_promote(tensor_a->type, tensor_b->type);

    php_ort_status_flow(
        !promotion.is_valid,
        {
            ort_math_broadcast_free(binfo);
            return NULL;
        },
        php_ort_status_math_invalidtype_ce,
        "%s: incompatible tensor types",
        operation_name
    );

    ort_tensor_t* result = ort_math_result_tensor(
        binfo->result_shape, binfo->result_dimensions, 
        promotion.result_type, operation_name);

    /* Parallel slow path using thread pool */
    size_t total = ort_math_result_total(
        binfo->result_shape, binfo->result_dimensions);
    size_t pool_size = ort_pool_cores();
    size_t chunk = (total + pool_size - 1) / pool_size;
    size_t num_chunks = (total + chunk - 1) / chunk;

    ort_pool_slow_binary_ctx_t ctx = {
        .layout = {
            .element = php_ort_type_sizeof(promotion.result_type),
            .total = total,
            .chunk = chunk
        },
        .result = result->data,
        .a = tensor_a->data,
        .b = tensor_b->data,
        .result_shape = binfo->result_shape,
        .result_dimensions = binfo->result_dimensions,
        .a_shape = tensor_a->shape,
        .a_dimensions = tensor_a->dimensions,
        .b_shape = tensor_b->shape,
        .b_dimensions = tensor_b->dimensions,
        .op = operation,
        .a_type = tensor_a->type,
        .b_type = tensor_b->type,
        .result_type = promotion.result_type,
        .element_size = php_ort_type_sizeof(promotion.result_type),
        .a_element_size = php_ort_type_sizeof(tensor_a->type),
        .b_element_size = php_ort_type_sizeof(tensor_b->type),
        .a_dim_offset = binfo->result_dimensions - tensor_a->dimensions,
        .b_dim_offset = binfo->result_dimensions - tensor_b->dimensions
    };

    ort_pool_submit(
        ort_pool_slow_binary_worker, &ctx, num_chunks);
    ort_math_broadcast_free(binfo);
    return result;
}

ort_tensor_t* ort_math_result_element_wise_scalar(
    ort_tensor_t* tensor,
    zval* scalar,
    ort_math_scalar_op_func_t operation,
    const char* operation_name
) {
    /* Create result tensor with same shape and type as input */
    ort_tensor_t* result = ort_math_result_tensor(
        tensor->shape,
        tensor->dimensions,
        tensor->type, operation_name);
    
    /* Convert scalar to appropriate type */
    void* scalar_data = NULL;
    float scalar_float;
    double scalar_double;
    int8_t scalar_int8;
    int16_t scalar_int16;
    int32_t scalar_int32;
    int64_t scalar_int64;
    uint8_t scalar_uint8;
    uint16_t scalar_uint16;
    uint32_t scalar_uint32;
    uint8_t scalar_bool;
    
    switch (tensor->type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            scalar_float = 
                (Z_TYPE_P(scalar) == IS_DOUBLE) ?
                    (float)Z_DVAL_P(scalar) :
                        (float)Z_LVAL_P(scalar);
            scalar_data = &scalar_float;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            scalar_double =
                (Z_TYPE_P(scalar) == IS_DOUBLE) ?
                    Z_DVAL_P(scalar) :
                        (double)Z_LVAL_P(scalar);
            scalar_data = &scalar_double;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            scalar_int8 =
                (Z_TYPE_P(scalar) == IS_LONG) ?
                    (int8_t)Z_LVAL_P(scalar) :
                        (int8_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_int8;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            scalar_int16 =
                (Z_TYPE_P(scalar) == IS_LONG) ?
                    (int16_t)Z_LVAL_P(scalar) :
                        (int16_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_int16;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            scalar_int32 =
                (Z_TYPE_P(scalar) == IS_LONG) ?
                    (int32_t)Z_LVAL_P(scalar) :
                        (int32_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_int32;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            scalar_int64 =
                (Z_TYPE_P(scalar) == IS_LONG) ?
                    Z_LVAL_P(scalar) :
                        (int64_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_int64;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            scalar_uint8 =
                (Z_TYPE_P(scalar) == IS_LONG) ?
                    (uint8_t)Z_LVAL_P(scalar) :
                        (uint8_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_uint8;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            scalar_uint16 =
                (Z_TYPE_P(scalar) == IS_LONG) ?
                    (uint16_t)Z_LVAL_P(scalar) :
                        (uint16_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_uint16;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            scalar_uint32 =
                (Z_TYPE_P(scalar) == IS_LONG) ?
                    (uint32_t)Z_LVAL_P(scalar) :
                        (uint32_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_uint32;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            scalar_bool =
                (Z_TYPE_P(scalar) == IS_LONG) ?
                    (uint8_t)Z_LVAL_P(scalar) :
                        (uint8_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_bool;
            break;

        default:
            php_ort_status_throw(php_ort_status_math_invalidtype_ce,
                "%s: unsupported tensor type for scalar operation",
                operation_name);
            ort_tensor_release(result);
            return NULL;
    }

    /* Parallelize scalar operation */
    size_t elements = tensor->elements;
    size_t element_size = php_ort_type_sizeof(tensor->type);
    size_t pool_size = ort_pool_cores();
    size_t chunk = (elements + pool_size - 1) / pool_size;
    size_t num_chunks = (elements + chunk - 1) / chunk;

    ort_pool_scalar_ctx_t ctx = {
        .layout = {
            .element = element_size,
            .total = elements,
            .chunk = chunk
        },
        .result = result->data,
        .a = tensor->data,
        .b = scalar_data,
        .op = operation
    };

    ort_pool_submit(ort_pool_scalar_worker, &ctx, num_chunks);
    return result;
}

ort_tensor_t* ort_math_result_element_wise_unary(
    ort_tensor_t* tensor,
    ort_math_unary_op_func_t operation,
    const char* operation_name
) {
    /* Create result tensor with same shape and type as input */
    ort_tensor_t* result = ort_math_result_tensor(
        tensor->shape,
        tensor->dimensions,
        tensor->type, operation_name);
    
    if (!result) {
        return NULL;
    }

    size_t elements = tensor->elements;
    size_t element_size = php_ort_type_sizeof(tensor->type);
    size_t pool_size = ort_pool_cores();
    size_t chunk = (elements + pool_size - 1) / pool_size;
    size_t num_chunks = (elements + chunk - 1) / chunk;

    ort_pool_unary_ctx_t ctx = {
        .layout = {
            .element = element_size,
            .total = elements,
            .chunk = chunk
        },
        .result = result->data,
        .a = tensor->data,
        .op = operation
    };

    ort_pool_submit(ort_pool_unary_worker, &ctx, num_chunks);
    return result;
}

/* Shape manipulation helpers */
ort_tensor_t* ort_math_result_tensor(
    const int64_t* shape,
    size_t dimensions,
    ONNXTensorElementDataType type,
    const char* name_prefix
) {
    ort_tensor_t* tensor = pecalloc(1, sizeof(ort_tensor_t), 0);
    
    tensor->refcount = 1;
    tensor->owner = PHP_ORT_OWN_ZEND;
    tensor->type = type;
    tensor->dimensions = dimensions;
    
    /* Copy shape */
    if (dimensions > 0 && shape != NULL) {
        tensor->shape = pecalloc(dimensions, sizeof(int64_t), 0);
        memcpy(tensor->shape, shape, dimensions * sizeof(int64_t));
    } else {
        tensor->shape = NULL;
    }
    
    /* Calculate total elements */
    if (dimensions > 0 && shape != NULL) {
        tensor->elements = ort_math_result_total(shape, dimensions);
    } else {
        tensor->elements = 1; // Scalar tensor has 1 element
    }
    
    /* Allocate data */
    size_t element_size = php_ort_tensor_sizeof(tensor);
    tensor->data = pecalloc(tensor->elements, element_size, 0);
    
    /* Generate name */
    char name_buffer[64];
    snprintf(name_buffer,
        sizeof(name_buffer),
        "%s_result_%p",
        name_prefix, tensor);

    tensor->name = zend_string_init(
        name_buffer, strlen(name_buffer), 0);

    return tensor;
}