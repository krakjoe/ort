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

/* Stack allocation threshold for math optimizations */
#define ORT_MATH_RESULT_STACK_DIMENSIONS 8

/* Result management */
ort_math_result_t* ort_math_result_create(ort_tensor_t* tensor) {
    ort_math_result_t* result = ecalloc(1, sizeof(ort_math_result_t));
    if (!result) {
        return NULL;
    }
    memset(result, 0, sizeof(ort_math_result_t));
    result->tensor = tensor;
    result->success = 1;
    result->error_message = NULL;
    return result;
}

void ort_math_result_free(ort_math_result_t* result) {
    if (result) {
        if (result->error_message) {
            zend_string_release(result->error_message);
        }
        efree(result);
    }
}

/* Utility functions */
size_t ort_math_result_total(const int64_t* shape, size_t dimensions) {
    size_t total = 1;
    for (size_t i = 0; i < dimensions; i++) {
        total *= shape[i];
    }
    return total;
}

zend_long ort_math_result_flat(const int64_t* indices, const int64_t* shape, size_t dimensions) {
    zend_long flat_index = 0;
    zend_long stride = 1;
    
    /* Calculate flat index using row-major order */
    for (size_t i = dimensions; i > 0; i--) {
        flat_index += indices[i-1] * stride;
        stride *= shape[i-1];
    }
    
    return flat_index;
}

void ort_math_result_multi(zend_long flat_index, const int64_t* shape, size_t dimensions, int64_t* indices) {
    /* Convert flat index back to multi-dimensional indices */
    for (size_t i = dimensions; i > 0; i--) {
        indices[i-1] = flat_index % shape[i-1];
        flat_index /= shape[i-1];
    }
}

/* Element-wise operation helpers implementation */
ort_math_result_t* ort_math_result_element_wise_binary(
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
    if (!binfo->is_compatible) {
        ort_math_broadcast_free(binfo);
        php_ort_status_throw(php_ort_status_math_invalidshape_ce,
            "%s: tensor shapes are not broadcast compatible", operation_name);
        return NULL;
    }

    /* Type promotion */
    ort_math_type_promotion_t promotion = ort_math_type_promote(tensor_a->type, tensor_b->type);
    if (!promotion.is_valid) {
        ort_math_broadcast_free(binfo);
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,
            "%s: incompatible tensor types", operation_name);
        return NULL;
    }

    ort_tensor_t* result_tensor = ort_math_result_tensor(
        binfo->result_shape, binfo->result_dimensions, promotion.result_type, operation_name);
    if (!result_tensor) {
        ort_math_broadcast_free(binfo);
        return NULL;
    }

    /* Broadcasting index logic with stack allocation optimization */
    size_t total = ort_math_result_total(binfo->result_shape, binfo->result_dimensions);
    
    /* Stack allocation for small tensors (< 8 dimensions) */
    int64_t out_indices_stack[ORT_MATH_RESULT_STACK_DIMENSIONS];
    int64_t a_indices_stack[ORT_MATH_RESULT_STACK_DIMENSIONS];
    int64_t b_indices_stack[ORT_MATH_RESULT_STACK_DIMENSIONS];
    
    int64_t* out_indices;
    int64_t* a_indices;
    int64_t* b_indices;
    zend_bool use_stack = (binfo->result_dimensions <= ORT_MATH_RESULT_STACK_DIMENSIONS);
    
    if (use_stack) {
        /* Use stack allocation for small tensors - no heap overhead */
        out_indices = out_indices_stack;
        a_indices = a_indices_stack;
        b_indices = b_indices_stack;
    } else {
        /* Fall back to heap allocation for large tensors */
        out_indices = ecalloc(binfo->result_dimensions, sizeof(int64_t));
        a_indices = ecalloc(binfo->result_dimensions, sizeof(int64_t));
        b_indices = ecalloc(binfo->result_dimensions, sizeof(int64_t));
        
        memset(out_indices, 0, binfo->result_dimensions * sizeof(int64_t));
        memset(a_indices, 0, binfo->result_dimensions * sizeof(int64_t));
        memset(b_indices, 0, binfo->result_dimensions * sizeof(int64_t));
    }
    for (size_t flat = 0; flat < total; flat++) {
        ort_math_result_multi(
            flat, 
            binfo->result_shape,
            binfo->result_dimensions,
            out_indices);

        // Map output indices to a/b indices
        size_t a_dim_offset = binfo->result_dimensions - tensor_a->dimensions;
        size_t b_dim_offset = binfo->result_dimensions - tensor_b->dimensions;
        
        for (size_t i = 0; i < binfo->result_dimensions; i++) {
            a_indices[i] = (i < a_dim_offset) ? 0 :
                (tensor_a->shape[i - a_dim_offset] == 1 ? 0 : out_indices[i]);
            b_indices[i] = (i < b_dim_offset) ? 0 :
                (tensor_b->shape[i - b_dim_offset] == 1 ? 0 : out_indices[i]);
        }

        zend_long a_flat = ort_math_result_flat(
            a_indices + a_dim_offset,
            tensor_a->shape, 
            tensor_a->dimensions);

        zend_long b_flat = ort_math_result_flat(
            b_indices + b_dim_offset,
            tensor_b->shape,
            tensor_b->dimensions);
        
        // Type promotion: operate on promoted type
        ort_tensor_t temp_tensor = {0};
        temp_tensor.type = promotion.result_type;
        void* res_ptr = 
            (char*)result_tensor->data + 
                flat * php_ort_tensor_sizeof(&temp_tensor);

        ort_tensor_t temp_tensor_a = {0};
        temp_tensor_a.type = tensor_a->type;
        void* a_ptr = 
            (char*)tensor_a->data + 
                a_flat * php_ort_tensor_sizeof(&temp_tensor_a);

        ort_tensor_t temp_tensor_b = {0};
        temp_tensor_b.type = tensor_b->type;
        void* b_ptr = 
            (char*)tensor_b->data + 
            b_flat * php_ort_tensor_sizeof(&temp_tensor_b);

        // Use a temporary buffer for type promotion if needed
        char a_buf[16], b_buf[16];
        ort_math_cast_element(
            a_ptr, a_buf, tensor_a->type, 
            promotion.result_type);
        ort_math_cast_element(
            b_ptr, b_buf, tensor_b->type,
            promotion.result_type);
        operation(res_ptr, a_buf, b_buf, 1);
    }

    /* Clean up heap-allocated index arrays if needed */
    if (!use_stack) {
        if (out_indices) efree(out_indices);
        if (a_indices) efree(a_indices);
        if (b_indices) efree(b_indices);
    }

    ort_math_broadcast_free(binfo);

    ort_math_result_t* result = ort_math_result_create(result_tensor);
    
    return result;
}

ort_math_result_t* ort_math_result_element_wise_binary_fast(
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b,
    ort_math_element_op_func_t operation,
    const char* operation_name
) {
    /* Create result tensor with same shape and type as inputs */
    ort_tensor_t* result_tensor = ort_math_result_tensor(
        tensor_a->shape, 
        tensor_a->dimensions,
        tensor_a->type, operation_name);
    
    if (!result_tensor) {
        return NULL;
    }

    operation(
        result_tensor->data,
        tensor_a->data,
        tensor_b->data,
        tensor_a->elements);

    ort_math_result_t* result = ort_math_result_create(result_tensor);
    
    return result;
}

ort_math_result_t* ort_math_result_element_wise_scalar(
    ort_tensor_t* tensor,
    zval* scalar,
    ort_math_scalar_op_func_t operation,
    const char* operation_name
) {
    /* Create result tensor with same shape and type as input */
    ort_tensor_t* result_tensor = ort_math_result_tensor(
        tensor->shape,
        tensor->dimensions,
        tensor->type, operation_name);
    
    if (!result_tensor) {
        return NULL;
    }
    
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
            return NULL;
    }
    
    /* Perform operation */
    operation(
        result_tensor->data,
        tensor->data,
        scalar_data,
        tensor->elements);
    
    ort_math_result_t* result = ort_math_result_create(result_tensor);
    
    return result;
}

ort_math_result_t* ort_math_result_element_wise_unary(
    ort_tensor_t* tensor,
    ort_math_unary_op_func_t operation,
    const char* operation_name
) {
    /* Create result tensor with same shape and type as input */
    ort_tensor_t* result_tensor = ort_math_result_tensor(
        tensor->shape,
        tensor->dimensions,
        tensor->type, operation_name);
    
    if (!result_tensor) {
        return NULL;
    }

    operation(
        result_tensor->data,
        tensor->data,
        tensor->elements);
    
    ort_math_result_t* result = ort_math_result_create(result_tensor);
    
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