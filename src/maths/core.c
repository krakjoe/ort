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

#include "ort.h"

#include "status.h"
#include "tensor.h"

#include "maths/core.h"
#include "maths/frontend.h"

#ifdef ORT_SIMD_ENABLED
#include "maths/simd/impl.h"
#endif

/* Type casting macros for element-wise ops */
/* Define source type macro block with all destination cases */
#define ORT_CAST_SOURCE_BLOCK(SRC_TYPE, SRC_CTYPE) \
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_##SRC_TYPE: { \
        SRC_CTYPE src_val = *(SRC_CTYPE*)src; \
        switch (dst_type) { \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   *(float*)dst = (float)src_val; break; \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  *(double*)dst = (double)src_val; break; \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    *(int8_t*)dst = (int8_t)src_val; break; \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   *(int16_t*)dst = (int16_t)src_val; break; \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   *(int32_t*)dst = (int32_t)src_val; break; \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   *(int64_t*)dst = (int64_t)src_val; break; \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   *(uint8_t*)dst = (uint8_t)src_val; break; \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  *(uint16_t*)dst = (uint16_t)src_val; break; \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  *(uint32_t*)dst = (uint32_t)src_val; break; \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    *(uint8_t*)dst = (src_val != 0) ? 1 : 0; break; \
            default: break; \
        } \
    } break;

/* Type casting helper for element-wise ops */
void ort_math_cast_element(const void* src, void* dst, ONNXTensorElementDataType src_type, ONNXTensorElementDataType dst_type) {
    /* Fast path for same type */
    if (src_type == dst_type) {
        ort_tensor_t temp_tensor = {0};
        temp_tensor.type = src_type;
        size_t size = php_ort_tensor_sizeof(&temp_tensor);
        if (size == 0) {
            return; /* Unsupported type */
        }
        memcpy(dst, src, size);
        return;
    }

    /* Handle type conversions using macros */
    switch (src_type) {
        ORT_CAST_SOURCE_BLOCK(FLOAT, float)
        ORT_CAST_SOURCE_BLOCK(DOUBLE, double)
        ORT_CAST_SOURCE_BLOCK(INT8, int8_t)
        ORT_CAST_SOURCE_BLOCK(INT16, int16_t)
        ORT_CAST_SOURCE_BLOCK(INT32, int32_t)
        ORT_CAST_SOURCE_BLOCK(INT64, int64_t)
        ORT_CAST_SOURCE_BLOCK(UINT8, uint8_t)
        ORT_CAST_SOURCE_BLOCK(UINT16, uint16_t)
        ORT_CAST_SOURCE_BLOCK(UINT32, uint32_t)
        
        /* Handle BOOL source type */
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            uint8_t src_val = *(uint8_t*)src;
            switch (dst_type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   *(float*)dst = (float)src_val; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  *(double*)dst = (double)src_val; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    *(int8_t*)dst = (int8_t)src_val; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   *(int16_t*)dst = (int16_t)src_val; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   *(int32_t*)dst = (int32_t)src_val; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   *(int64_t*)dst = (int64_t)src_val; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   *(uint8_t*)dst = (uint8_t)src_val; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  *(uint16_t*)dst = (uint16_t)src_val; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  *(uint32_t*)dst = (uint32_t)src_val; break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    *(uint8_t*)dst = src_val; break;
                default: break;
            }
        } break;
        
        default:
            break;
    }
}

/* Broadcasting implementation */
ort_math_broadcast_info_t* ort_math_broadcast_calculate(
    const int64_t* shape_a, size_t dims_a,
    const int64_t* shape_b, size_t dims_b
) {
    ort_math_broadcast_info_t* info = ecalloc(1, sizeof(ort_math_broadcast_info_t));
    if (!info) {
        return NULL;
    }
    memset(info, 0, sizeof(ort_math_broadcast_info_t));
    
    /* Determine result dimensions (max of both) */
    info->result_dimensions = dims_a > dims_b ? dims_a : dims_b;
    info->result_shape = ecalloc(info->result_dimensions, sizeof(int64_t));
    info->is_compatible = 1;
    info->needs_broadcast_a = 0;
    info->needs_broadcast_b = 0;
    
    /* Work backwards from the last dimension */
    for (size_t i = 0; i < info->result_dimensions; i++) {
        size_t idx_a = dims_a > i ? dims_a - 1 - i : SIZE_MAX;
        size_t idx_b = dims_b > i ? dims_b - 1 - i : SIZE_MAX;
        size_t result_idx = info->result_dimensions - 1 - i;
        
        int64_t dim_a = (idx_a != SIZE_MAX) ? shape_a[idx_a] : 1;
        int64_t dim_b = (idx_b != SIZE_MAX) ? shape_b[idx_b] : 1;
        
        if (dim_a == dim_b) {
            info->result_shape[result_idx] = dim_a;
        } else if (dim_a == 1) {
            info->result_shape[result_idx] = dim_b;
            info->needs_broadcast_a = 1;
        } else if (dim_b == 1) {
            info->result_shape[result_idx] = dim_a; 
            info->needs_broadcast_b = 1;
        } else {
            /* Incompatible shapes */
            info->is_compatible = 0;
            break;
        }
    }
    
    return info;
}

void ort_math_broadcast_free(ort_math_broadcast_info_t* info) {
    if (info) {
        if (info->result_shape) {
            efree(info->result_shape);
        }
        efree(info);
    }
}

/* Helper function to check if a type is a signed integer */
static zend_always_inline int ort_math_is_signed_integer(ONNXTensorElementDataType type) {
    return (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
            type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 ||
            type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
            type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
}

/* Helper function to check if a type is an unsigned integer */
static zend_always_inline int ort_math_is_unsigned_integer(ONNXTensorElementDataType type) {
    return (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
            type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ||
            type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32);
}

/* Helper function to get the larger signed integer type */
static zend_always_inline ONNXTensorElementDataType ort_math_promote_signed_integers(ONNXTensorElementDataType type_a, ONNXTensorElementDataType type_b) {
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
static zend_always_inline ONNXTensorElementDataType ort_math_promote_unsigned_integers(ONNXTensorElementDataType type_a, ONNXTensorElementDataType type_b) {
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
    ort_math_type_promotion_t result = {ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, 0};
    /* ...existing code for permissive promotion... */
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
    else if (ort_math_is_signed_integer(type_a) && ort_math_is_signed_integer(type_b)) {
        result.result_type = ort_math_promote_signed_integers(type_a, type_b);
        result.is_valid = 1;
    }
    /* Handle unsigned integer types */
    else if (ort_math_is_unsigned_integer(type_a) && ort_math_is_unsigned_integer(type_b)) {
        result.result_type = ort_math_promote_unsigned_integers(type_a, type_b);
        result.is_valid = 1;
    }
    /* Mixed signed/unsigned - promote to INT64 for safety */
    else if ((ort_math_is_signed_integer(type_a) && ort_math_is_unsigned_integer(type_b)) ||
             (ort_math_is_unsigned_integer(type_a) && ort_math_is_signed_integer(type_b))) {
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
    ort_math_type_promotion_t result = {ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, 0};

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
    else if (ort_math_is_signed_integer(type_a) && ort_math_is_signed_integer(type_b)) {
        result.result_type = ort_math_promote_signed_integers(type_a, type_b);
        result.is_valid = 1;
    }
    /* Handle unsigned integer types */
    else if (ort_math_is_unsigned_integer(type_a) && ort_math_is_unsigned_integer(type_b)) {
        result.result_type = ort_math_promote_unsigned_integers(type_a, type_b);
        result.is_valid = 1;
    }
    /* Mixed signed/unsigned - follow ONNX/NumPy promotion rules */
    else if ((ort_math_is_signed_integer(type_a) && ort_math_is_unsigned_integer(type_b)) ||
             (ort_math_is_unsigned_integer(type_a) && ort_math_is_signed_integer(type_b))) {
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

/* Free result structure */
void ort_math_result_destroy(ort_math_result_t* result) {
    if (result) {
        efree(result);
    }
}

/* Utility functions */
size_t ort_math_util_calculate_total_elements(const int64_t* shape, size_t dimensions) {
    size_t total = 1;
    for (size_t i = 0; i < dimensions; i++) {
        total *= shape[i];
    }
    return total;
}

zend_long ort_math_util_get_flat_index(const int64_t* indices, const int64_t* shape, size_t dimensions) {
    zend_long flat_index = 0;
    zend_long stride = 1;
    
    /* Calculate flat index using row-major order */
    for (size_t i = dimensions; i > 0; i--) {
        flat_index += indices[i-1] * stride;
        stride *= shape[i-1];
    }
    
    return flat_index;
}

void ort_math_util_get_multi_index(zend_long flat_index, const int64_t* shape, size_t dimensions, int64_t* indices) {
    /* Convert flat index back to multi-dimensional indices */
    for (size_t i = dimensions; i > 0; i--) {
        indices[i-1] = flat_index % shape[i-1];
        flat_index /= shape[i-1];
    }
}

/* Validation functions */
zend_bool ort_math_validate_input(ort_tensor_t* tensor, const char* operation_name) {
    if (!tensor || !tensor->data) {
        php_ort_status_flow(!SUCCESS, return 0, php_ort_status_tensor_invaliddata_ce,
            "%s: tensor has no data", operation_name);
    }

    if (tensor->elements == 0) {
        php_ort_status_flow(!SUCCESS, return 0, php_ort_status_tensor_invalidshape_ce,
            "%s: tensor is empty", operation_name);
    }

    /* Check for unsupported UINT64 */
    if (tensor->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64) {
        php_ort_status_flow(!SUCCESS, return 0, php_ort_status_tensor_invalidtype_ce,
            "%s: UINT64 tensor type is not supported", operation_name);
    }
    
    return 1;
}

zend_bool ort_math_validate_axis(ort_tensor_t* tensor, zend_long axis, const char* operation_name) {
    if (axis < 0) {
        axis += tensor->dimensions;
    }

    if (axis < 0 || axis >= tensor->dimensions) {
        php_ort_status_flow(!SUCCESS, return 0, php_ort_status_tensor_invalidshape_ce,
            "%s: axis %ld is out of bounds for tensor with %zu dimensions", 
            operation_name, axis, tensor->dimensions);
    }

    return 1;
}

/* Element-wise operation helpers implementation */
ort_math_result_t* ort_math_result_element_wise_binary(
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b,
    ort_math_element_op_func_t operation,
    const char* operation_name
) {
    /* Fast path: identical shapes and types */
    if (ort_math_shapes_identical(tensor_a, tensor_b)) {
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

    ort_tensor_t* result_tensor = ort_tensor_create_result(
        binfo->result_shape, binfo->result_dimensions, promotion.result_type, operation_name);
    if (!result_tensor) {
        ort_math_broadcast_free(binfo);
        return NULL;
    }

    /* Broadcasting index logic with stack allocation optimization */
    size_t total = ort_math_util_calculate_total_elements(binfo->result_shape, binfo->result_dimensions);
    
    /* Stack allocation for small tensors (< 8 dimensions) */
    int64_t out_indices_stack[ORT_MATH_MAX_STACK_DIMENSIONS];
    int64_t a_indices_stack[ORT_MATH_MAX_STACK_DIMENSIONS];
    int64_t b_indices_stack[ORT_MATH_MAX_STACK_DIMENSIONS];
    
    int64_t* out_indices;
    int64_t* a_indices;
    int64_t* b_indices;
    zend_bool use_stack = (binfo->result_dimensions <= ORT_MATH_MAX_STACK_DIMENSIONS);
    
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
        if (!out_indices || !a_indices || !b_indices) {
            return NULL; /* Memory allocation failed */
        }
        memset(out_indices, 0, binfo->result_dimensions * sizeof(int64_t));
        memset(a_indices, 0, binfo->result_dimensions * sizeof(int64_t));
        memset(b_indices, 0, binfo->result_dimensions * sizeof(int64_t));
    }
    for (size_t flat = 0; flat < total; flat++) {
        ort_math_util_get_multi_index(flat, binfo->result_shape, binfo->result_dimensions, out_indices);
        // Map output indices to a/b indices
        size_t a_dim_offset = binfo->result_dimensions - tensor_a->dimensions;
        size_t b_dim_offset = binfo->result_dimensions - tensor_b->dimensions;
        for (size_t i = 0; i < binfo->result_dimensions; i++) {
            a_indices[i] = (i < a_dim_offset) ? 0 :
                (tensor_a->shape[i - a_dim_offset] == 1 ? 0 : out_indices[i]);
            b_indices[i] = (i < b_dim_offset) ? 0 :
                (tensor_b->shape[i - b_dim_offset] == 1 ? 0 : out_indices[i]);
        }
        zend_long a_flat = ort_math_util_get_flat_index(a_indices + a_dim_offset, tensor_a->shape, tensor_a->dimensions);
        zend_long b_flat = ort_math_util_get_flat_index(b_indices + b_dim_offset, tensor_b->shape, tensor_b->dimensions);
        // Type promotion: operate on promoted type
        ort_tensor_t temp_tensor = {0};
        temp_tensor.type = promotion.result_type;
        void* res_ptr = (char*)result_tensor->data + flat * php_ort_tensor_sizeof(&temp_tensor);
        
        ort_tensor_t temp_tensor_a = {0};
        temp_tensor_a.type = tensor_a->type;
        void* a_ptr = (char*)tensor_a->data + a_flat * php_ort_tensor_sizeof(&temp_tensor_a);
        
        ort_tensor_t temp_tensor_b = {0};
        temp_tensor_b.type = tensor_b->type;
        void* b_ptr = (char*)tensor_b->data + b_flat * php_ort_tensor_sizeof(&temp_tensor_b);
        // Use a temporary buffer for type promotion if needed
        char a_buf[16], b_buf[16];
        ort_math_cast_element(a_ptr, a_buf, tensor_a->type, promotion.result_type);
        ort_math_cast_element(b_ptr, b_buf, tensor_b->type, promotion.result_type);
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
    ort_tensor_t* result_tensor = ort_tensor_create_result(
        tensor_a->shape, tensor_a->dimensions, tensor_a->type, operation_name);
    
    if (!result_tensor) {
        return NULL;
    }

    operation(result_tensor->data, tensor_a->data, tensor_b->data, tensor_a->elements);

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
    ort_tensor_t* result_tensor = ort_tensor_create_result(
        tensor->shape, tensor->dimensions, tensor->type, operation_name);
    
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
            scalar_float = (Z_TYPE_P(scalar) == IS_DOUBLE) ? (float)Z_DVAL_P(scalar) : (float)Z_LVAL_P(scalar);
            scalar_data = &scalar_float;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            scalar_double = (Z_TYPE_P(scalar) == IS_DOUBLE) ? Z_DVAL_P(scalar) : (double)Z_LVAL_P(scalar);
            scalar_data = &scalar_double;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            scalar_int8 = (Z_TYPE_P(scalar) == IS_LONG) ? (int8_t)Z_LVAL_P(scalar) : (int8_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_int8;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            scalar_int16 = (Z_TYPE_P(scalar) == IS_LONG) ? (int16_t)Z_LVAL_P(scalar) : (int16_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_int16;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            scalar_int32 = (Z_TYPE_P(scalar) == IS_LONG) ? (int32_t)Z_LVAL_P(scalar) : (int32_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_int32;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            scalar_int64 = (Z_TYPE_P(scalar) == IS_LONG) ? Z_LVAL_P(scalar) : (int64_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_int64;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            scalar_uint8 = (Z_TYPE_P(scalar) == IS_LONG) ? (uint8_t)Z_LVAL_P(scalar) : (uint8_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_uint8;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            scalar_uint16 = (Z_TYPE_P(scalar) == IS_LONG) ? (uint16_t)Z_LVAL_P(scalar) : (uint16_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_uint16;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            scalar_uint32 = (Z_TYPE_P(scalar) == IS_LONG) ? (uint32_t)Z_LVAL_P(scalar) : (uint32_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_uint32;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            scalar_bool = (Z_TYPE_P(scalar) == IS_LONG) ? (uint8_t)Z_LVAL_P(scalar) : (uint8_t)Z_DVAL_P(scalar);
            scalar_data = &scalar_bool;
            break;
        default:
            php_ort_status_throw(php_ort_status_math_invalidtype_ce,
                "%s: unsupported tensor type for scalar operation", operation_name);
            return NULL;
    }
    
    /* Perform operation */
    operation(result_tensor->data, tensor->data, scalar_data, tensor->elements);
    
    ort_math_result_t* result = ort_math_result_create(result_tensor);
    
    return result;
}

ort_math_result_t* ort_math_result_element_wise_unary(
    ort_tensor_t* tensor,
    ort_math_unary_op_func_t operation,
    const char* operation_name
) {
    /* Create result tensor with same shape and type as input */
    ort_tensor_t* result_tensor = ort_tensor_create_result(
        tensor->shape, tensor->dimensions, tensor->type, operation_name);
    
    if (!result_tensor) {
        return NULL;
    }

    operation(result_tensor->data, tensor->data, tensor->elements);
    
    ort_math_result_t* result = ort_math_result_create(result_tensor);
    
    return result;
}

/* Shape manipulation helpers */
ort_tensor_t* ort_tensor_create_result(
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
        tensor->elements = ort_math_util_calculate_total_elements(shape, dimensions);
    } else {
        tensor->elements = 1; // Scalar tensor has 1 element
    }
    
    /* Allocate data */
    size_t element_size = php_ort_tensor_sizeof(tensor);
    tensor->data = pecalloc(tensor->elements, element_size, 0);
    
    /* Generate name */
    char name_buffer[64];
    snprintf(name_buffer, sizeof(name_buffer), "%s_result_%p", name_prefix, tensor);
    tensor->name = zend_string_init(name_buffer, strlen(name_buffer), 0);
    
    return tensor;
}

/* Shape comparison helper for fast path optimization */
zend_bool ort_math_shapes_identical(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b) {
    /* Quick checks: dimensions and type must match */
    if (tensor_a->dimensions != tensor_b->dimensions || tensor_a->type != tensor_b->type) {
        return 0;
    }
    
    /* Compare each dimension */
    for (size_t i = 0; i < tensor_a->dimensions; i++) {
        if (tensor_a->shape[i] != tensor_b->shape[i]) {
            return 0;
        }
    }

    return 1;
}

/* Complete dispatch table with SIMD optimization */
static ort_math_type_dispatch_t ort_math_dispatch_table[] = {
    /* FLOAT */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        .add_func        = ort_math_ops_add_float,
        .sub_func        = ort_math_ops_sub_float,
        .mul_func        = ort_math_ops_mul_float,
        .div_func        = ort_math_ops_div_float,
        .mod_func        = ort_math_ops_mod_float,
        .pow_func        = ort_math_ops_pow_float,
        .ceil_func       = ort_math_ops_ceil_float,
        .floor_func      = ort_math_ops_floor_float,
        .round_func      = ort_math_ops_round_float,
        .abs_func        = ort_math_ops_abs_float,
        .sqrt_func       = ort_math_ops_sqrt_float,
        .neg_func        = ort_math_ops_neg_float,
        .recip_func      = ort_math_ops_recip_float,
        .sign_func       = ort_math_ops_sign_float,
        .trunc_func      = ort_math_ops_trunc_float,
        .dot_func        = ort_math_ops_dot_float,
    },
    /* DOUBLE */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        .add_func        = ort_math_ops_add_double,
        .sub_func        = ort_math_ops_sub_double,
        .mul_func        = ort_math_ops_mul_double,
        .div_func        = ort_math_ops_div_double,
        .mod_func        = ort_math_ops_mod_double,
        .pow_func        = ort_math_ops_pow_double,
        .ceil_func       = ort_math_ops_ceil_double,
        .floor_func      = ort_math_ops_floor_double,
        .round_func      = ort_math_ops_round_double,
        .abs_func        = ort_math_ops_abs_double,
        .sqrt_func       = ort_math_ops_sqrt_double,
        .neg_func        = ort_math_ops_neg_double,
        .recip_func      = ort_math_ops_recip_double,
        .sign_func       = ort_math_ops_sign_double,
        .trunc_func      = ort_math_ops_trunc_double,
        .dot_func        = ort_math_ops_dot_double,
    },
    /* INT8 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        .add_func        = ort_math_ops_add_int8_t,
        .sub_func        = ort_math_ops_sub_int8_t,
        .mul_func        = ort_math_ops_mul_int8_t,
        .div_func        = ort_math_ops_div_int8_t,
        .mod_func        = ort_math_ops_mod_int8_t,
        .pow_func        = ort_math_ops_pow_int8_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_int8_t,
        .neg_func        = ort_math_ops_neg_int8_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_int8_t,
    },
    /* INT16 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
        .add_func        = ort_math_ops_add_int16_t,
        .sub_func        = ort_math_ops_sub_int16_t,
        .mul_func        = ort_math_ops_mul_int16_t,
        .div_func        = ort_math_ops_div_int16_t,
        .mod_func        = ort_math_ops_mod_int16_t,
        .pow_func        = ort_math_ops_pow_int16_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_int16_t,
        .neg_func        = ort_math_ops_neg_int16_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_int16_t,
    },
    /* INT32 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        .add_func        = ort_math_ops_add_int32_t,
        .sub_func        = ort_math_ops_sub_int32_t,
        .mul_func        = ort_math_ops_mul_int32_t,
        .div_func        = ort_math_ops_div_int32_t,
        .mod_func        = ort_math_ops_mod_int32_t,
        .pow_func        = ort_math_ops_pow_int32_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_int32_t,
        .neg_func        = ort_math_ops_neg_int32_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_int32_t,
    },
    /* INT64 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        .add_func        = ort_math_ops_add_int64_t,
        .sub_func        = ort_math_ops_sub_int64_t,
        .mul_func        = ort_math_ops_mul_int64_t,
        .div_func        = ort_math_ops_div_int64_t,
        .mod_func        = ort_math_ops_mod_int64_t,
        .pow_func        = ort_math_ops_pow_int64_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_int64_t,
        .neg_func        = ort_math_ops_neg_int64_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_int64_t,
    },
    /* UINT8 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        .add_func        = ort_math_ops_add_uint8_t,
        .sub_func        = ort_math_ops_sub_uint8_t,
        .mul_func        = ort_math_ops_mul_uint8_t,
        .div_func        = ort_math_ops_div_uint8_t,
        .mod_func        = ort_math_ops_mod_uint8_t,
        .pow_func        = ort_math_ops_pow_uint8_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_uint8_t,
        .neg_func        = ort_math_ops_neg_uint8_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_uint8_t,
    },
    /* UINT16 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
        .add_func        = ort_math_ops_add_uint16_t,
        .sub_func        = ort_math_ops_sub_uint16_t,
        .mul_func        = ort_math_ops_mul_uint16_t,
        .div_func        = ort_math_ops_div_uint16_t,
        .mod_func        = ort_math_ops_mod_uint16_t,
        .pow_func        = ort_math_ops_pow_uint16_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_uint16_t,
        .neg_func        = ort_math_ops_neg_uint16_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_uint16_t,
    },
    /* UINT32 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
        .add_func        = ort_math_ops_add_uint32_t,
        .sub_func        = ort_math_ops_sub_uint32_t,
        .mul_func        = ort_math_ops_mul_uint32_t,
        .div_func        = ort_math_ops_div_uint32_t,
        .mod_func        = ort_math_ops_mod_uint32_t,
        .pow_func        = ort_math_ops_pow_uint32_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_uint32_t,
        .neg_func        = ort_math_ops_neg_uint32_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_uint32_t,
    },
    /* BOOL */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
        .add_func        = ort_math_ops_add_zend_bool,   // Logical OR (or addition)
        .sub_func        = ort_math_ops_sub_zend_bool,   // Logical XOR (or subtraction)
        .mul_func        = ort_math_ops_mul_zend_bool,   // Logical AND (or multiplication)
        .div_func        = ort_math_ops_div_zend_bool,   // Logical division (A && B)
        .mod_func        = NULL,                         // Logical modulo (A && !B)
        .pow_func        = NULL,                         // Logical power (A ** B)
        .ceil_func       = NULL,                         // Not meaningful for bool
        .floor_func      = NULL,                         // Not meaningful for bool
        .round_func      = NULL,                         // Not meaningful for bool
        .abs_func        = NULL,                         // Identity for bool
        .sqrt_func       = NULL,                         // Not meaningful for bool
        .neg_func        = ort_math_ops_neg_zend_bool,   // Logical NOT
        .recip_func      = NULL,                         // Not meaningful for bool
        .sign_func       = NULL,                         // Identity for bool
        .trunc_func      = NULL,                         // Not meaningful for bool
        .dot_func        = NULL,
    }
};

/* Get dispatch table entry for a given type */
const ort_math_type_dispatch_t* ort_math_get_dispatch(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return &ort_math_dispatch_table[0];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return &ort_math_dispatch_table[1];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return &ort_math_dispatch_table[2];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return &ort_math_dispatch_table[3];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return &ort_math_dispatch_table[4];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return &ort_math_dispatch_table[5];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return &ort_math_dispatch_table[6];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return &ort_math_dispatch_table[7];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return &ort_math_dispatch_table[8];
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return &ort_math_dispatch_table[9];
        default:
            return NULL;
    }
}

void ort_math_startup() {
#ifdef ORT_SIMD_ENABLED
    ort_math_simd_install(
        ort_math_dispatch_table);
#endif
}

void ort_math_shutdown() {
    // Currently no specific shutdown logic, but can be extended in the future
}