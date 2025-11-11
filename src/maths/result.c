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
#include "maths/backend/impl.h"
#include "maths/broadcast.h"
#include "maths/cast.h"
#include "maths/dispatch.h"
#include "maths/promotion.h"
#include "maths/result.h"
#include "maths/validate.h"
#include "maths/pool.h"

static zend_always_inline ort_tensor_t* 
    ort_math_result_element_wise_binary_fast(
    ort_math_promotion_t* promotion,
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b,
    ort_math_kernel_binary_t kernel,
    const char* operator
) {
    /* Create result tensor with same shape and type as inputs */
    ort_tensor_t* result = ort_math_result_tensor(
        tensor_a->shape, 
        tensor_a->dimensions,
        promotion ? 
            promotion->result_type :
                tensor_a->type, operator);

    void* a_buf = ort_math_operation_upcast(result, promotion, tensor_a->data);
    void* b_buf = ort_math_operation_upcast(result, promotion, tensor_b->data);

#ifdef ORT_BACKEND_GPU_ENABLED
    if (ORT_MATH_DISPATCH_TAGGED(kernel, GPU)) {
        ort_math_kernel_binary_t gpu =
            ort_math_backend_gpu_kernel(
                kernel, result->type,
                    result, 2, a_buf, b_buf);
        if (gpu) {
            ((ort_math_kernel_binary_t)
                ORT_MATH_DISPATCH_UNTAG(gpu))(
                    result->data,
                    a_buf,
                    b_buf,
                    result->elements);
            goto __ort_math_result_element_wise_binary_fast_cleanup;
        }
    }
#endif

    /* Chunk work for pool */
    size_t chunk;
    size_t num_chunks = ort_pool_chunk(
        tensor_a->elements,
        php_ort_type_sizeof(
            result->type), &chunk);

    ort_pool_binary_ctx_t ctx = {
        .layout = {
            .element = php_ort_type_sizeof(result->type),
            .total = tensor_a->elements,
            .chunk = chunk
        },
        .result = result->data,
        .a = a_buf,
        .b = b_buf,
        .op = ORT_MATH_DISPATCH_UNTAG(kernel)
    };

    ort_pool_submit(ort_pool_binary_worker, &ctx, num_chunks);

__ort_math_result_element_wise_binary_fast_cleanup:
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
    ort_math_promotion_t* promotion,
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b,
    ort_math_kernel_binary_t kernel,
    const char* operator
) {
    /* Fast path: identical shapes and types */
    if (ort_math_validate_identity(tensor_a, tensor_b)) {
        return ort_math_result_element_wise_binary_fast(
            promotion, tensor_a, tensor_b, kernel, operator);
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
        operator
    );

    /* Create result tensor */
    ort_tensor_t* result = ort_math_result_tensor(
        binfo->result_shape, binfo->result_dimensions,
        promotion->result_type, operator);

    /* Upcast and broadcast both inputs to result buffer */
    void* a_buf = ort_alloc(
        php_ort_type_sizeof(promotion->result_type),
        result->elements);
    void* b_buf = ort_alloc(
        php_ort_type_sizeof(promotion->result_type),
        result->elements);
    ort_math_broadcast_perform(
        result,
        tensor_a, a_buf);
    ort_math_broadcast_perform(
        result,
        tensor_b, b_buf);

#ifdef ORT_BACKEND_GPU_ENABLED
    if (ORT_MATH_DISPATCH_TAGGED(kernel, GPU)) {
        ort_math_kernel_binary_t gpu =
            ort_math_backend_gpu_kernel(
                kernel, result->type,
                    result, 2, a_buf, b_buf);
        if (gpu) {
            ((ort_math_kernel_binary_t)
                ORT_MATH_DISPATCH_UNTAG(gpu))(
                    result->data,
                    a_buf,
                    b_buf,
                    result->elements);
            goto __ort_math_result_element_wise_binary_cleanup;
        }
    }
#endif

    /* Chunk work for pool */
    size_t chunk;
    size_t num_chunks = ort_pool_chunk(
        result->elements,
        php_ort_type_sizeof(
            promotion->result_type), &chunk);

    ort_pool_binary_ctx_t ctx = {
        .layout = {
            .element = php_ort_type_sizeof(promotion->result_type),
            .total = result->elements,
            .chunk = chunk
        },
        .result = result->data,
        .a = a_buf,
        .b = b_buf,
        .op = ORT_MATH_DISPATCH_UNTAG(kernel)
    };

    ort_pool_submit(ort_pool_binary_worker, &ctx, num_chunks);

__ort_math_result_element_wise_binary_cleanup:
    ort_free(a_buf);
    ort_free(b_buf);
    ort_math_broadcast_free(binfo);
    return result;
}

ort_tensor_t* ort_math_result_element_wise_scalar(
    ort_math_promotion_t* promotion,
    ort_tensor_t* tensor,
    zval* scalar,
    ort_math_kernel_scalar_t kernel,
    const char* operator
) {
    /* Create result tensor with same shape and promoted type as input */
    ort_tensor_t* result = ort_math_result_tensor(
        tensor->shape,
        tensor->dimensions,
        promotion ?
            promotion->result_type : tensor->type, operator);

    /* Upcast input if needed */
    void* a_buf = ort_math_operation_upcast(result, promotion, tensor->data);

    /* Convert scalar to promoted type */
    uint8_t scalar_buffer[16];
    void* scalar_data = scalar_buffer;
    switch (result->type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            *(float16*)scalar_data = (Z_TYPE_P(scalar) == IS_DOUBLE) ?
                ort_math_float16_from_float64((float64)Z_DVAL_P(scalar)) :
                ort_math_float16_from_float64((float64)Z_LVAL_P(scalar));
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32:
            *(float32*)scalar_data = (Z_TYPE_P(scalar) == IS_DOUBLE) ? (float32)Z_DVAL_P(scalar) : (float32)Z_LVAL_P(scalar);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64:
            *(float64*)scalar_data = (Z_TYPE_P(scalar) == IS_DOUBLE) ? Z_DVAL_P(scalar) : (float64)Z_LVAL_P(scalar);
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
                operator);
            ort_tensor_release(result);
            if (a_buf != tensor->data) 
                ort_free(a_buf);
            return NULL;
    }

    /* Chunk work for pool */
    size_t chunk;
    size_t num_chunks = ort_pool_chunk(
        tensor->elements,
        php_ort_type_sizeof(result->type), &chunk);

    ort_pool_scalar_ctx_t ctx = {
        .layout = {
            .element = php_ort_type_sizeof(result->type),
            .total = tensor->elements,
            .chunk = chunk
        },
        .result = result->data,
        .a = a_buf,
        .b = scalar_data,
        .op = ORT_MATH_DISPATCH_UNTAG(kernel)
    };

    ort_pool_submit(ort_pool_scalar_worker, &ctx, num_chunks);
    if (a_buf != tensor->data)
        ort_free(a_buf);
    return result;
}

ort_tensor_t* ort_math_result_element_wise_unary(
    ort_math_promotion_t* promotion,
    ort_tensor_t* tensor,
    ort_math_kernel_unary_t kernel,
    const char* operator
) {
    /* Create result tensor with same shape and promoted type as input */
    ort_tensor_t* result = ort_math_result_tensor(
        tensor->shape,
        tensor->dimensions,
        promotion ? promotion->result_type : tensor->type, 
        operator);
    if (!result) {
        return NULL;
    }

    /* Upcast input if needed */
    void* a_buf = ort_math_operation_upcast(result, promotion, tensor->data);

#ifdef ORT_BACKEND_GPU_ENABLED
    if (ORT_MATH_DISPATCH_TAGGED(kernel, GPU)) {
        ort_math_kernel_unary_t gpu =
            ort_math_backend_gpu_kernel(
                kernel, result->type,
                    result, 1, a_buf);
        if (gpu) {
            ((ort_math_kernel_unary_t)
                ORT_MATH_DISPATCH_UNTAG(gpu))(
                    result->data,
                    a_buf,
                    result->elements);
            goto __ort_math_result_element_wise_unary_cleanup;
        }
    }
#endif

    /* Chunk work for pool */
    size_t chunk;
    size_t num_chunks = ort_pool_chunk(
        tensor->elements,
        php_ort_type_sizeof(
            result->type), &chunk);

    ort_pool_unary_ctx_t ctx = {
        .layout = {
            .element = php_ort_type_sizeof(result->type),
            .total = tensor->elements,
            .chunk = chunk
        },
        .result = result->data,
        .a = a_buf,
        .op = ORT_MATH_DISPATCH_UNTAG(kernel)
    };

    ort_pool_submit(ort_pool_unary_worker, &ctx, num_chunks);

__ort_math_result_element_wise_unary_cleanup:
    if (promotion && promotion->upcast.count) {
        ort_free((void*)a_buf);
    }
    return result;
}

ort_tensor_t* ort_math_result_serial_element_wise_reduce_tensor(
    ort_math_promotion_t *promotion,
    ort_tensor_t* tensor,
    ort_math_kernel_reduce_tensor_t kernel,
    const char* operator
) {
    // Output is always a scalar (0-dim tensor)
    ort_tensor_t* result = ort_math_result_tensor(
        NULL, 0,
        promotion ?
            promotion->result_type :
                tensor->type,
        operator);
    if (!result) {
        return NULL;
    }

    void* buffer = ort_math_operation_upcast(result, promotion, tensor->data);

    /* Call the operation directly, no threading */
    ((ort_math_kernel_reduce_tensor_t)
        ORT_MATH_DISPATCH_UNTAG(kernel))(
            result->data, buffer, tensor->elements);

    /* Free upcasted buffer if necessary */
    if (promotion && promotion->upcast.count) {
        ort_free((void*)buffer);
    }

    return result;
}

ort_tensor_t* ort_math_result_element_wise_reduce_tensor(
    ort_math_promotion_t *promotion,
    ort_tensor_t* tensor,
    ort_math_kernel_reduce_tensor_t kernel,
    const char* operator
) {
    // Output is always a scalar (0-dim tensor)
    ort_tensor_t* result = ort_math_result_tensor(
        NULL, 0,
        promotion ?
            promotion->result_type :
                tensor->type,
        operator);
    if (!result) {
        return NULL;
    }

    ort_pool_reduce_tensor_ctx_t ctx = {
        .layout = {
            .element = php_ort_type_sizeof(result->type),
            .total   = 1, // Only one output (scalar)
            .chunk   = 1
        },
        .result   = result->data,
        .a        = ort_math_operation_upcast(result, promotion, tensor->data),
        .elements = tensor->elements,
        .op       = ORT_MATH_DISPATCH_UNTAG(kernel)
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
    ort_math_promotion_t* promotion,
    ort_tensor_t* tensor,
    size_t axis,
    zend_bool keepdims,
    ort_math_kernel_reduce_axis_t kernel,
    const char* operator,
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
        operator);
    efree(result_shape);

    /* Chunk work for pool */
    size_t chunk;
    size_t num_chunks = ort_pool_chunk(
        result->elements,
        php_ort_type_sizeof(result->type), &chunk);

    ort_pool_reduce_axis_ctx_t ctx = {
        .layout = {
            .element = php_ort_type_sizeof(result->type),
            .total   = result->elements,
            .chunk   = chunk
        },
        .result = result->data,
        .a      = ort_math_operation_upcast(result, promotion, tensor->data),
        .op     = ORT_MATH_DISPATCH_UNTAG(kernel),
        .input_shape = tensor->shape,
        .input_dims = tensor->dimensions,
        .output_shape = result->shape,
        .output_dims = result->dimensions,
        .axis = axis
    };

    ort_pool_submit(ort_pool_reduce_axis_worker, &ctx, num_chunks);

    if (promotion && promotion->upcast.count) {
        ort_free((void*)ctx.a);
    }

    return result;
}

ort_tensor_t* ort_math_result_serial_element_wise_reduce_axis(
    ort_math_promotion_t* promotion,
    ort_tensor_t* tensor,
    size_t axis,
    zend_bool keepdims,
    ort_math_kernel_reduce_axis_t kernel,
    const char* operator,
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
        operator);
    efree(result_shape);

    void* buffer = ort_math_operation_upcast(result, promotion, tensor->data);

    ((ort_math_kernel_reduce_axis_t)
        ORT_MATH_DISPATCH_UNTAG(kernel))(
            result->data, buffer,
            tensor->shape, tensor->dimensions,
            result->shape, result->dimensions,
            axis);

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

    /* Allocate data */
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