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
#include "alloc.h"
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
    ort_math_type_promotion_t* promotion,
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b,
    ort_math_element_op_func_t operation,
    const char* operation_name
) {
    /* Create result tensor with same shape and type as inputs */
    ort_tensor_t* result = ort_math_result_tensor(
        tensor_a->shape, 
        tensor_a->dimensions,
        promotion ? 
            promotion->result_type :
                tensor_a->type, operation_name);

    size_t elements = tensor_a->elements;
    size_t element_size = php_ort_type_sizeof(result->type);
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
        .a = ort_math_operation_upcast(result, promotion, tensor_a->data),
        .b = ort_math_operation_upcast(result, promotion, tensor_b->data),
        .op = operation
    };

    ort_pool_submit(ort_pool_binary_worker, &ctx, num_chunks);

    if (promotion && promotion->upcast.count) {
        for (size_t i = 0; i < promotion->upcast.count; ++i) {
            if (promotion->upcast.inputs[i] &&
                promotion->upcast.inputs[i]->data == ctx.a) {
                ort_free((void*)ctx.a);
            } else if (promotion->upcast.inputs[i] &&
                       promotion->upcast.inputs[i]->data == ctx.b) {
                ort_free((void*)ctx.b);
            }
        }
    }

    return result;
}

/* Element-wise operation helpers implementation */
ort_tensor_t* ort_math_result_element_wise_binary(
    ort_math_type_promotion_t* promotion,
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b,
    ort_math_element_op_func_t operation,
    const char* operation_name
) {
    /* Fast path: identical shapes and types */
    if (ort_math_validate_identity(tensor_a, tensor_b)) {
        return ort_math_result_element_wise_binary_fast(
            promotion, tensor_a, tensor_b, operation, operation_name);
    }

    /* General path: use centralized upcast/broadcast logic */
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

    /* Create result tensor */
    ort_tensor_t* result = ort_math_result_tensor(
        binfo->result_shape, binfo->result_dimensions,
        promotion->result_type, operation_name);

    /* Upcast and broadcast both inputs to result buffer */
    void* a_buf = ort_alloc(
        php_ort_type_sizeof(promotion->result_type),
        result->elements);
    void* b_buf = ort_alloc(
        php_ort_type_sizeof(promotion->result_type),
        result->elements);
    ort_math_operation_broadcast(
        result,
        promotion,
        tensor_a, a_buf);
    ort_math_operation_broadcast(
        result,
        promotion,
        tensor_b, b_buf);

    /* Use fast worker for the upcasted, broadcasted buffers */
    size_t pool_size = ort_pool_cores();
    size_t chunk = (result->elements + pool_size - 1) / pool_size;
    size_t num_chunks = (result->elements + chunk - 1) / chunk;

    ort_pool_binary_ctx_t ctx = {
        .layout = {
            .element = php_ort_type_sizeof(promotion->result_type),
            .total = result->elements,
            .chunk = chunk
        },
        .result = result->data,
        .a = a_buf,
        .b = b_buf,
        .op = operation
    };

    ort_pool_submit(ort_pool_binary_worker, &ctx, num_chunks);

    ort_free(a_buf);
    ort_free(b_buf);
    ort_math_broadcast_free(binfo);
    return result;
}

ort_tensor_t* ort_math_result_element_wise_scalar(
    ort_math_type_promotion_t* promotion,
    ort_tensor_t* tensor,
    zval* scalar,
    ort_math_scalar_op_func_t operation,
    const char* operation_name
) {
    /* Create result tensor with same shape and promoted type as input */
    ort_tensor_t* result = ort_math_result_tensor(
        tensor->shape,
        tensor->dimensions,
        promotion ? promotion->result_type : tensor->type, operation_name);

    /* Upcast input if needed */
    void* a_buf = ort_math_operation_upcast(result, promotion, tensor->data);

    /* Convert scalar to promoted type */
    uint8_t scalar_buffer[16];
    void* scalar_data = scalar_buffer;
    switch (result->type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            *(float*)scalar_data = (Z_TYPE_P(scalar) == IS_DOUBLE) ? (float)Z_DVAL_P(scalar) : (float)Z_LVAL_P(scalar);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            *(double*)scalar_data = (Z_TYPE_P(scalar) == IS_DOUBLE) ? Z_DVAL_P(scalar) : (double)Z_LVAL_P(scalar);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            *(int8_t*)scalar_data = (Z_TYPE_P(scalar) == IS_LONG) ? (int8_t)Z_LVAL_P(scalar) : (int8_t)Z_DVAL_P(scalar);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            *(int16_t*)scalar_data = (Z_TYPE_P(scalar) == IS_LONG) ? (int16_t)Z_LVAL_P(scalar) : (int16_t)Z_DVAL_P(scalar);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            *(int32_t*)scalar_data = (Z_TYPE_P(scalar) == IS_LONG) ? (int32_t)Z_LVAL_P(scalar) : (int32_t)Z_DVAL_P(scalar);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            *(int64_t*)scalar_data = (Z_TYPE_P(scalar) == IS_LONG) ? Z_LVAL_P(scalar) : (int64_t)Z_DVAL_P(scalar);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            *(uint8_t*)scalar_data = (Z_TYPE_P(scalar) == IS_LONG) ? (uint8_t)Z_LVAL_P(scalar) : (uint8_t)Z_DVAL_P(scalar);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            *(uint16_t*)scalar_data = (Z_TYPE_P(scalar) == IS_LONG) ? (uint16_t)Z_LVAL_P(scalar) : (uint16_t)Z_DVAL_P(scalar);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            *(uint32_t*)scalar_data = (Z_TYPE_P(scalar) == IS_LONG) ? (uint32_t)Z_LVAL_P(scalar) : (uint32_t)Z_DVAL_P(scalar);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            *(uint8_t*)scalar_data = (Z_TYPE_P(scalar) == IS_LONG) ? (uint8_t)Z_LVAL_P(scalar) : (uint8_t)Z_DVAL_P(scalar);
            break;
        default:
            php_ort_status_throw(php_ort_status_math_invalidtype_ce,
                "%s: unsupported tensor type for scalar operation",
                operation_name);
            ort_tensor_release(result);
            if (a_buf != tensor->data) 
                ort_free(a_buf);
            return NULL;
    }

    /* Parallelize scalar operation */
    size_t elements = tensor->elements;
    size_t element_size = php_ort_type_sizeof(result->type);
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
        .a = a_buf,
        .b = scalar_data,
        .op = operation
    };

    ort_pool_submit(ort_pool_scalar_worker, &ctx, num_chunks);
    if (a_buf != tensor->data)
        ort_free(a_buf);
    return result;
}

ort_tensor_t* ort_math_result_element_wise_unary(
    ort_math_type_promotion_t* promotion,
    ort_tensor_t* tensor,
    ort_math_unary_op_func_t operation,
    const char* operation_name
) {
    /* Create result tensor with same shape and promoted type as input */
    ort_tensor_t* result = ort_math_result_tensor(
        tensor->shape,
        tensor->dimensions,
        promotion ? promotion->result_type : tensor->type, 
        operation_name);
    if (!result) {
        return NULL;
    }

    /* Upcast input if needed */
    void* a_buf = ort_math_operation_upcast(result, promotion, tensor->data);

    size_t elements = tensor->elements;
    size_t element_size = php_ort_type_sizeof(result->type);
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
        .a = a_buf,
        .op = operation
    };

    ort_pool_submit(ort_pool_unary_worker, &ctx, num_chunks);
    if (promotion && promotion->upcast.count) {
        ort_free((void*)a_buf);
    }
    return result;
}

ort_tensor_t* ort_math_result_serial_element_wise_reduce_tensor(
    ort_math_type_promotion_t *promotion,
    ort_tensor_t* tensor,
    void (*operation)(void *result, const void *a, size_t n),
    const char* operation_name
) {
    // Output is always a scalar (0-dim tensor)
    ort_tensor_t* result = ort_math_result_tensor(
        NULL, 0,
        promotion ?
            promotion->result_type :
                tensor->type,
        operation_name);
    if (!result) {
        return NULL;
    }

    void* buffer = ort_math_operation_upcast(result, promotion, tensor->data);

    /* Call the operation directly, no threading */
    operation(result->data, buffer, tensor->elements);

    /* Free upcasted buffer if necessary */
    if (promotion && promotion->upcast.count) {
        ort_free((void*)buffer);
    }

    return result;
}

ort_tensor_t* ort_math_result_element_wise_reduce_tensor(
    ort_math_type_promotion_t *promotion,
    ort_tensor_t* tensor,
    void (*operation)(void *result, const void *a, size_t n),
    const char* operation_name
) {
    // Output is always a scalar (0-dim tensor)
    ort_tensor_t* result = ort_math_result_tensor(
        NULL, 0,
        promotion ?
            promotion->result_type :
                tensor->type,
        operation_name);
    if (!result) {
        return NULL;
    }
    size_t elements = tensor->elements;
    size_t element_size = php_ort_type_sizeof(
        promotion ?
            promotion->result_type :
                tensor->type);
    
    ort_pool_reduce_tensor_ctx_t ctx = {
        .layout = {
            .element = element_size,
            .total   = 1, // Only one output (scalar)
            .chunk   = 1
        },
        .result   = result->data,
        .a        = ort_math_operation_upcast(result, promotion, tensor->data),
        .elements = elements,
        .op       = operation
    };

    // Only one chunk/thread needed for scalar reduction
    ort_pool_submit(ort_pool_reduce_tensor_worker, &ctx, 1);

    /* Free upcasted buffer if necessary */
    if (promotion && promotion->upcast.count) {
        ort_free((void*)ctx.a);
    }

    return result;
}

ort_tensor_t* ort_math_result_element_wise_reduce_axis(
    ort_math_type_promotion_t* promotion,
    ort_tensor_t* tensor,
    size_t axis,
    zend_bool keepdims,
    void (*operation)(void *result, const void *a, size_t outer, size_t as, size_t inner),
    const char* operation_name,
    int64_t* (*shape)(ort_tensor_t* tensor, size_t axis, zend_bool keepdims, size_t* result_dims)
) {
    // Calculate output shape
    size_t result_dims = 0;
    int64_t* result_shape = shape(
        tensor, axis, keepdims, &result_dims);

    // Create result tensor
    ort_tensor_t* result = ort_math_result_tensor(
        result_shape, result_dims,
        promotion ?
            promotion->result_type :
                tensor->type,
        operation_name);
    efree(result_shape);

    // Compute reduction layout (robust for all axis values)
    size_t outer = 1, inner = 1;
    if (axis > 0) {
        for (size_t i = 0; i < axis; ++i)
            outer *= tensor->shape[i];
    }
    if (axis + 1 < tensor->dimensions) {
        for (size_t i = axis + 1; i < tensor->dimensions; ++i)
            inner *= tensor->shape[i];
    }

    size_t axis_size    = tensor->shape[axis];
    size_t total        = outer * inner;
    size_t element_size = php_ort_type_sizeof(
        promotion ? promotion->result_type : tensor->type);
    size_t pool_size    = ort_pool_cores();
    size_t chunk        = (total + pool_size - 1) / pool_size;
    size_t num_chunks   = (total + chunk - 1) / chunk;

    ort_pool_reduce_axis_ctx_t ctx = {
        .layout = {
            .element = element_size,
            .total   = total,
            .chunk   = chunk,
            .axis_size = axis_size,
            .outer = outer,
            .inner = inner
        },
        .result = result->data,
        .a      = ort_math_operation_upcast(result, promotion, tensor->data),
        .op     = operation
    };

    ort_pool_submit(ort_pool_reduce_axis_worker, &ctx, num_chunks);

    if (promotion && promotion->upcast.count) {
        ort_free((void*)ctx.a);
    }

    return result;
}

ort_tensor_t* ort_math_result_serial_element_wise_reduce_axis(
    ort_math_type_promotion_t* promotion,
    ort_tensor_t* tensor,
    size_t axis,
    zend_bool keepdims,
    void (*operation)(void *result, const void *a, size_t outer, size_t as, size_t inner),
    const char* operation_name,
    int64_t* (*shape)(ort_tensor_t* tensor, size_t axis, zend_bool keepdims, size_t* result_dims)
) {
    // Calculate output shape
    size_t result_dims = 0;
    int64_t* result_shape = shape(
        tensor, axis, keepdims, &result_dims);

    // Create result tensor
    ort_tensor_t* result = ort_math_result_tensor(
        result_shape, result_dims,
        promotion ?
            promotion->result_type :
                tensor->type,
        operation_name);
    efree(result_shape);

    // Compute reduction layout (robust for all axis values)
    size_t outer = 1, inner = 1;
    if (axis > 0) {
        for (size_t i = 0; i < axis; ++i)
            outer *= tensor->shape[i];
    }
    if (axis + 1 < tensor->dimensions) {
        for (size_t i = axis + 1; i < tensor->dimensions; ++i)
            inner *= tensor->shape[i];
    }

    void* buffer = ort_math_operation_upcast(result, promotion, tensor->data);

    operation(
        result->data, buffer, outer, axis, inner);

    if (promotion && promotion->upcast.count) {
        ort_free((void*)buffer);
    }

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
    
    /* Allocate data */;
    tensor->data = ort_alloc(
        php_ort_tensor_sizeof(tensor),
        tensor->elements);
    
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