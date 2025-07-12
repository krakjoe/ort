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

#include "alloc.h"
#include "maths/promotion.h"
#include "maths/cast.h"
#include "maths/result.h"
#include "maths/validate.h"

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

static zend_always_inline int ort_math_type_promotion_schema_lookup(const ONNXTensorElementDataType* list, int size, ONNXTensorElementDataType type) {
    for (int i = 0; i < size; ++i) {
        if (list[i] == type) {
            return i;
        }
    }

    return -1;
}

// Get binary promotion type from schema
static zend_always_inline ONNXTensorElementDataType ort_math_type_promotion_schema_resolve_binary(
    const ort_math_type_promotion_schema_t* schema,
    ONNXTensorElementDataType a_type,
    ONNXTensorElementDataType b_type
) {
    int i = ort_math_type_promotion_schema_lookup(
        schema->indices, schema->size, a_type);
    int j = ort_math_type_promotion_schema_lookup(
        schema->indices, schema->size, b_type);
    
    if (i < 0 || j < 0)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

    return schema->table[i * schema->size + j];
}

// Get unary promotion type from schema
static zend_always_inline ONNXTensorElementDataType ort_math_type_promotion_schema_resolve_unary(
    const ort_math_type_promotion_schema_t* schema,
    ONNXTensorElementDataType t_type
) {
    int i = ort_math_type_promotion_schema_lookup(
        schema->indices, schema->size, t_type);
    
    if (i < 0)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

    return schema->table[i];
}

// Improved: Map PHP scalar to tensor dtype for int/float/bool, else fallback
static zend_always_inline ONNXTensorElementDataType ort_math_type_promotion_schema_resolve_zend(ONNXTensorElementDataType tensor_type, zend_long zend_type) {
    switch (zend_type) {
        case IS_TRUE:
        case IS_FALSE:
            // Only allow BOOL if tensor is BOOL
            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
            }
            // Otherwise, treat as int (NumPy: bool+int -> int)
            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) {
                return tensor_type;
            }
            // Otherwise, treat as float
            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
                return tensor_type;
            }
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        case IS_LONG:
            // If tensor is BOOL, cast to BOOL (NumPy: int + bool -> bool)
            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
            }
            // Use tensor's dtype if it's an integer type
            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) {
                return tensor_type;
            }
            // If tensor is float, treat as float
            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
                return tensor_type;
            }
            // Fallback
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case IS_DOUBLE:
            // Use tensor's dtype if it's a float type
            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
                return tensor_type;
            }
            // If tensor is integer, promote to float
            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) {
                // Use float if available, else double
                if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
            }
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        default:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

// Get scalar promotion type from schema
static zend_always_inline ONNXTensorElementDataType 
    ort_math_type_promotion_schema_resolve_scalar(
    const ort_math_type_promotion_schema_t* schema,
    ONNXTensorElementDataType onnx_type,
    zend_long zend_type
) {
    int i = ort_math_type_promotion_schema_lookup(
        schema->indices, schema->size, onnx_type);
    int j = ort_math_type_promotion_schema_lookup(
        schema->indices, schema->size,
            ort_math_type_promotion_schema_resolve_zend(
                onnx_type, zend_type));
    
    if (i < 0 || j < 0)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

    return schema->table[i * schema->size + j];
}

ort_math_type_promotion_t ort_math_type_promote_schema_binary(
    const ort_math_type_promotion_schema_t* schema,
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b
) {
    ort_math_type_promotion_t promotion = {
        .result_type =
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
        .is_valid = 0,
        .upcast = {
            .inputs = { 
                NULL, NULL
            },
            .count = 0
        }
    };

    ONNXTensorElementDataType resolved =
        ort_math_type_promotion_schema_resolve_binary(
            schema, tensor_a->type, tensor_b->type);

    if (resolved != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        promotion.result_type = resolved;
        promotion.is_valid = 1;

        if (tensor_a->type != promotion.result_type) {
            promotion.upcast.inputs[0] = tensor_a;
            promotion.upcast.count++;
        }
        if (tensor_b->type != promotion.result_type) {
            promotion.upcast.inputs[1] = tensor_b;
            promotion.upcast.count++;
        }
    }

    return promotion;
}

ort_math_type_promotion_t ort_math_type_promote_schema_unary(
    const ort_math_type_promotion_schema_t* schema,
    ort_tensor_t* tensor
) {
    ort_math_type_promotion_t promotion = {
        .result_type =
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
        .is_valid = 0,
        .upcast = {
            .inputs = { 
                NULL, NULL
            },
            .count = 0
        }
    };

    ONNXTensorElementDataType resolved =
        ort_math_type_promotion_schema_resolve_unary(schema, tensor->type);

    if (resolved != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        promotion.result_type = resolved;
        promotion.is_valid = 1;

        if (tensor->type != promotion.result_type) {
            promotion.upcast.inputs[0] = tensor;
            promotion.upcast.count++;
        }
    }

    return promotion;
}

ort_math_type_promotion_t ort_math_type_promote_schema_scalar(
    const ort_math_type_promotion_schema_t* schema,
    ort_tensor_t* tensor,
    zval* scalar
) {
    ort_math_type_promotion_t promotion = {
        .result_type =
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
        .is_valid = 0,
        .upcast = {
            .inputs = { 
                NULL, NULL
            },
            .count = 0
        }
    };

    ONNXTensorElementDataType resolved =
        ort_math_type_promotion_schema_resolve_scalar(
            schema, tensor->type, Z_TYPE_P(scalar));

    if (resolved != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        promotion.result_type = resolved;
        promotion.is_valid = 1;

        if (tensor->type != promotion.result_type) {
            promotion.upcast.inputs[0] = tensor;
            promotion.upcast.count++;
        }
    }

    return promotion;
}

/* Type promotion implementation (permissive, for elementwise ops) */
ort_math_type_promotion_t ort_math_type_promote(
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b
) {
    ort_math_type_promotion_t promotion = {
        .result_type =
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
        .is_valid = 0,
        .upcast = {
            .inputs = { 
                NULL, NULL
            },
            .count = 0
        }
    };

    /* Handle bool special cases first */
    if (tensor_a->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        promotion.result_type = tensor_b->type;
        promotion.is_valid = 1;
        goto __ort_math_type_promote_record;
    }
    if (tensor_b->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        promotion.result_type = tensor_a->type;
        promotion.is_valid = 1;
        goto __ort_math_type_promote_record;
    }
    /* Same types */
    if (tensor_a->type == tensor_b->type) {
        promotion.result_type = tensor_a->type;
        promotion.is_valid = 1;
        goto __ort_math_type_promote_record;
    }
    /* Type promotion hierarchy (simplified NumPy-style) */
    /* BOOL < INT8 < INT16 < INT32 < INT64 */
    /* UINT8 < UINT16 < UINT32 */  
    /* FLOAT < DOUBLE */
    /* Handle floating point types first */
    if (tensor_a->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE || tensor_b->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        promotion.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        promotion.is_valid = 1;
    } else if (tensor_a->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || tensor_b->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        promotion.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        promotion.is_valid = 1;
    }
    /* Handle signed integer types */
    else if (ort_math_promote_is_signed(tensor_a->type) &&
             ort_math_promote_is_signed(tensor_b->type)) {
        promotion.result_type = ort_math_promote_signed(tensor_a->type, tensor_b->type);
        promotion.is_valid = 1;
    }
    /* Handle unsigned integer types */
    else if (ort_math_promote_is_unsigned(tensor_a->type) &&
             ort_math_promote_is_unsigned(tensor_b->type)) {
        promotion.result_type = ort_math_promote_unsigned(tensor_a->type, tensor_b->type);
        promotion.is_valid = 1;
    }
    /* Mixed signed/unsigned - promote to INT64 for safety */
    else if ((ort_math_promote_is_signed(tensor_a->type)    &&
              ort_math_promote_is_unsigned(tensor_b->type)) ||
             (ort_math_promote_is_unsigned(tensor_a->type)  &&
              ort_math_promote_is_signed(tensor_b->type))) {
        promotion.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        promotion.is_valid = 1;
    }

__ort_math_type_promote_record:
    // Record upcasts if valid
    if (promotion.is_valid) {
        if (tensor_a->type != promotion.result_type) {
            promotion.upcast.inputs[0] = tensor_a;
            promotion.upcast.count++;
        }
        if (tensor_b->type != promotion.result_type) {
            promotion.upcast.inputs[1] = tensor_b;
            promotion.upcast.count++;
        }
    }

    return promotion;
}

/* Strict type promotion for dot/matmul/reductions (NumPy/ONNX style) */
ort_math_type_promotion_t ort_math_type_promote_strict(
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b
) {
    ort_math_type_promotion_t promotion = {
        .result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
        .is_valid    = 0,
        .upcast      = {
            .inputs = { NULL, NULL },
            .count = 0
        }
    };

    /* Handle bool special cases first */
    if (tensor_a->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        promotion.result_type = tensor_b->type;
        promotion.is_valid = 1;
    } else if (tensor_b->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        promotion.result_type = tensor_a->type;
        promotion.is_valid = 1;
    } else if (tensor_a->type == tensor_b->type) {
        promotion.result_type = tensor_a->type;
        promotion.is_valid = 1;
    } else if (tensor_a->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE || tensor_b->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        promotion.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        promotion.is_valid = 1;
    } else if (tensor_a->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || tensor_b->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        promotion.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        promotion.is_valid = 1;
    } else if (ort_math_promote_is_signed(tensor_a->type) &&
               ort_math_promote_is_signed(tensor_b->type)) {
        promotion.result_type = ort_math_promote_signed(tensor_a->type, tensor_b->type);
        promotion.is_valid = 1;
    } else if (ort_math_promote_is_unsigned(tensor_a->type) &&
               ort_math_promote_is_unsigned(tensor_b->type)) {
        promotion.result_type = ort_math_promote_unsigned(tensor_a->type, tensor_b->type);
        promotion.is_valid = 1;
    } else if ((ort_math_promote_is_signed(tensor_a->type) &&
                ort_math_promote_is_unsigned(tensor_b->type)) ||
               (ort_math_promote_is_unsigned(tensor_a->type) &&
                ort_math_promote_is_signed(tensor_b->type))) {
        // int8 + uint8 => int32
        if ((tensor_a->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 && tensor_b->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) ||
            (tensor_b->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 && tensor_a->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)) {
            promotion.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            promotion.is_valid = 1;
        }
        // int16 + uint16 => int32
        else if ((tensor_a->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 && tensor_b->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) ||
                 (tensor_b->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 && tensor_a->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
            promotion.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            promotion.is_valid = 1;
        }
        // int32 + uint32 => int64
        else if ((tensor_a->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && tensor_b->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32) ||
                 (tensor_b->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 && tensor_a->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)) {
            promotion.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            promotion.is_valid = 1;
        }
        // All other mixed signed/unsigned cases are not supported (e.g., int64 + uint32, etc.)
        else {
            promotion.result_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            promotion.is_valid = 0;
        }
    }

    // Record upcasts if valid
    if (promotion.is_valid) {
        if (tensor_a->type != promotion.result_type) {
            promotion.upcast.inputs[0] = tensor_a; // placeholder, to be set by caller
            promotion.upcast.count++;
        }
        if (tensor_b->type != promotion.result_type) {
            promotion.upcast.inputs[1] = tensor_b; // placeholder, to be set by caller
            promotion.upcast.count++;
        }
    }

    return promotion;
}

ort_math_type_promotion_t ort_math_operation_promote(
    ONNXTensorElementDataType type,
    size_t inputs,
    ... /* ort_tensor_t* inputs[] */
) {
    ort_math_type_promotion_t promotion = {
        .result_type =
            type,
        .is_valid = 1,
        .upcast = {
            .inputs = {
                NULL,
                NULL
            },
            .count = 0
        }
    };

    va_list args;
    va_start(args, inputs);
    for (size_t i = 0; i < inputs; i++) {
        assert(i < 2);

        ort_tensor_t* tensor =
            va_arg(args, ort_tensor_t*);
        
        if (type != tensor->type) {
            promotion.upcast.inputs[i] = tensor;
            promotion.upcast.count++;
        }
    }
    va_end(args);

    return promotion;
}

#define ORT_MATH_RESULT_STACK_DIMENSIONS 8

// Centralized upcast + broadcast routine for elementwise ops
void* ort_math_operation_broadcast(
    const ort_tensor_t* result,
    const ort_math_type_promotion_t* promotion,
    const ort_tensor_t* input,
    void* out_buf
) {
    // Broadcast input to result shape and upcast to result type
    int64_t out_indices[ORT_MATH_RESULT_STACK_DIMENSIONS];
    int64_t in_indices[ORT_MATH_RESULT_STACK_DIMENSIONS];
    size_t in_offset = result->dimensions - input->dimensions;
    size_t in_type_size = php_ort_type_sizeof(input->type);
    size_t out_type_size = php_ort_type_sizeof(promotion->result_type);
    for (size_t i = 0; i < result->elements; ++i) {
        ort_math_result_multi(i, result->shape, result->dimensions, out_indices);
        for (size_t d = 0; d < result->dimensions; ++d) {
            in_indices[d] = (d < in_offset) ? 0 :
                (input->shape[d - in_offset] == 1 ? 0 : out_indices[d]);
        }
        zend_long in_flat = ort_math_result_flat(
            in_indices + in_offset,
            input->shape, input->dimensions);
        void* src_ptr = (char*)input->data + in_flat * in_type_size;
        void* dst_ptr = (char*)out_buf + i * out_type_size;
        ort_math_cast_element(src_ptr, dst_ptr, input->type, promotion->result_type);
    }
    return out_buf;
}

void* ort_math_operation_upcast(
    const ort_tensor_t* result,
    const ort_math_type_promotion_t* promotion,
    void* data
) {
    /* Short circuit if no upcasting required */
    if (!promotion || promotion->upcast.count == 0) {
        return data; // No upcasting needed
    }

    for (size_t i = 0; i < promotion->upcast.count; i++) {
        if (promotion->upcast.inputs[i]->data == data) {

            void *cast = ort_alloc(
                promotion->upcast.inputs[i]->elements,
                php_ort_type_sizeof(
                    promotion->result_type
                )
            );

            ort_math_cast_buffer(
                promotion->upcast.inputs[i]->data,
                cast,
                promotion->upcast.inputs[i]->type,
                promotion->result_type,
                promotion->upcast.inputs[i]->elements
            );

            return cast;
        }
    }

    return data;
}