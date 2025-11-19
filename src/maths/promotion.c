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

/* {{{ schemas */
#include "maths/schema/abs.h"
#include "maths/schema/acos.h"
#include "maths/schema/arccos.h"
#include "maths/schema/arccosh.h"
#include "maths/schema/arcsin.h"
#include "maths/schema/arcsinh.h"
#include "maths/schema/arctan.h"
#include "maths/schema/arctanh.h"
#include "maths/schema/argmax.h"
#include "maths/schema/argmin.h"
#include "maths/schema/atan.h"
#include "maths/schema/add.h"
#include "maths/schema/asin.h"
#include "maths/schema/cbrt.h"
#include "maths/schema/ceil.h"
#include "maths/schema/cos.h"
#include "maths/schema/cosh.h"
#include "maths/schema/div.h"
#include "maths/schema/dot.h"
#include "maths/schema/exp.h"
#include "maths/schema/exp2.h"
#include "maths/schema/floor.h"
#include "maths/schema/log.h"
#include "maths/schema/log2.h"
#include "maths/schema/log10.h"
#include "maths/schema/matmul.h"
#include "maths/schema/max.h"
#include "maths/schema/mean.h"
#include "maths/schema/min.h"
#include "maths/schema/mod.h"
#include "maths/schema/mul.h"
#include "maths/schema/neg.h"
#include "maths/schema/pow.h"
#include "maths/schema/recip.h"
#include "maths/schema/round.h"
#include "maths/schema/sign.h"
#include "maths/schema/sin.h"
#include "maths/schema/sinh.h"
#include "maths/schema/softmax.h"
#include "maths/schema/sqrt.h"
#include "maths/schema/sub.h"
#include "maths/schema/sum.h"
#include "maths/schema/tan.h"
#include "maths/schema/tanh.h"
#include "maths/schema/trunc.h"
/* }}} */

static HashTable __ort_math_promotion_schema = {0};

static zend_always_inline int ort_math_promotion_schema_lookup(const ONNXTensorElementDataType* list, int size, ONNXTensorElementDataType type) {
    for (int i = 0; i < size; ++i) {
        if (list[i] == type) {
            return i;
        }
    }

    return -1;
}

// Get binary promotion type from schema
ONNXTensorElementDataType ort_math_promotion_resolve_binary(
    const ort_math_promotion_schema_t* schema,
    ONNXTensorElementDataType a_type,
    ONNXTensorElementDataType b_type
) {
    int i = ort_math_promotion_schema_lookup(
        schema->indices, schema->size, a_type);
    int j = ort_math_promotion_schema_lookup(
        schema->indices, schema->size, b_type);
    
    if (i < 0 || j < 0)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

    return schema->table[i * schema->size + j];
}

// Get unary promotion type from schema
ONNXTensorElementDataType ort_math_promotion_resolve_unary(
    const ort_math_promotion_schema_t* schema,
    ONNXTensorElementDataType t_type
) {
    int i = ort_math_promotion_schema_lookup(
        schema->indices, schema->size, t_type);
    
    if (i < 0)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

    return schema->table[i];
}

static zend_always_inline ONNXTensorElementDataType ort_math_promotion_schema_resolve_zend(ONNXTensorElementDataType tensor_type, zend_long zend_type) {
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
            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64) {
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
            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64) {
                return tensor_type;
            }
            // Fallback
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case IS_DOUBLE:
            // Use tensor's dtype if it's a float type
            if (tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32 ||
                tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64) {
                return tensor_type;
            }
            // Fallback
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64;
        default:
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

// Get scalar promotion type from schema
static zend_always_inline ONNXTensorElementDataType 
    ort_math_promotion_schema_resolve_scalar(
    const ort_math_promotion_schema_t* schema,
    ONNXTensorElementDataType onnx_type,
    zend_long zend_type
) {
    int i = ort_math_promotion_schema_lookup(
        schema->indices, schema->size, onnx_type);
    int j = ort_math_promotion_schema_lookup(
        schema->indices, schema->size,
            ort_math_promotion_schema_resolve_zend(
                onnx_type, zend_type));
    
    if (i < 0 || j < 0)
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

    return schema->table[i * schema->size + j];
}

ort_math_promotion_t ort_math_promotion_perform_binary(
    const ort_math_promotion_schema_t* schema,
    ort_tensor_t* tensor_a,
    ort_tensor_t* tensor_b
) {
    ort_math_promotion_t promotion = {
        .result_type =
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
        .inputs = {
            .lhs = tensor_a,
            .rhs = tensor_b,
        },
        .upcast = {
            .inputs = { 
                NULL, NULL
            },
            .count = 0
        },
    };

    ONNXTensorElementDataType resolved =
        ort_math_promotion_resolve_binary(
            schema, tensor_a->type, tensor_b->type);

    if (resolved != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        promotion.result_type = resolved;

        if (schema->operands == ORT_MATH_PROMOTION_SCHEMA_OPERANDS_PRESERVE) {
            return promotion;
        }

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

ort_math_promotion_t ort_math_promotion_perform_unary(
    const ort_math_promotion_schema_t* schema,
    ort_tensor_t* tensor
) {
    ort_math_promotion_t promotion = {
        .result_type =
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
        .inputs = {
            .lhs = tensor,
            .rhs = NULL,
        },
        .upcast = {
            .inputs = { 
                NULL, NULL
            },
            .count = 0
        }
    };

    ONNXTensorElementDataType resolved =
        ort_math_promotion_resolve_unary(schema, tensor->type);

    if (resolved != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        promotion.result_type = resolved;

        if (schema->operands == ORT_MATH_PROMOTION_SCHEMA_OPERANDS_PRESERVE) {
            return promotion;
        }

        if (tensor->type != promotion.result_type) {
            promotion.upcast.inputs[0] = tensor;
            promotion.upcast.count++;
        }
    }

    return promotion;
}

ort_math_promotion_t ort_math_promotion_perform_scalar(
    const ort_math_promotion_schema_t* schema,
    ort_tensor_t* tensor,
    zval* scalar
) {
    ort_math_promotion_t promotion = {
        .result_type =
            ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
        .inputs = {
            .lhs = tensor,
            .rhs = (void*) scalar,
        },
        .upcast = {
            .inputs = { 
                NULL, NULL
            },
            .count = 0
        }
    };

    ONNXTensorElementDataType resolved =
        ort_math_promotion_schema_resolve_scalar(
            schema, tensor->type, Z_TYPE_P(scalar));

    if (resolved != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        promotion.result_type = resolved;

        if (schema->operands == ORT_MATH_PROMOTION_SCHEMA_OPERANDS_PRESERVE) {
            return promotion;
        }

        if (tensor->type != promotion.result_type) {
            promotion.upcast.inputs[0] = tensor;
            promotion.upcast.count++;
        }
    }

    return promotion;
}

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-result"
#endif
char* ort_math_promotion_explain(
    const ort_math_promotion_t* promotion,
    const ort_math_promotion_schema_t* schema) {

    char *explain = NULL;

    switch (schema->kind) {
        case ORT_MATH_PROMOTION_SCHEMA_KIND_BINARY:
            asprintf(&explain, "%s, %s -> %s",
                php_ort_type_name(promotion->inputs.lhs->type),
                php_ort_type_name(promotion->inputs.rhs->type),
                php_ort_type_name(promotion->result_type));
            break;

        case ORT_MATH_PROMOTION_SCHEMA_KIND_UNARY:
            asprintf(&explain, "%s -> %s",
                php_ort_type_name(promotion->inputs.lhs->type),
                php_ort_type_name(promotion->result_type));
            break;

        default:
            /* unreachable */
            explain = strdup("Invalid promotion schema");
            break;
    }

    return explain;
}
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
void* ort_math_operation_upcast(
    const ort_tensor_t* result,
    const ort_math_promotion_t* promotion,
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

void ort_math_promotion_startup(void) {
    /* we know that the first caller of this
        (the main php process) is unable to concurrently
        re-enter until this call completes. 
       we know after that call (when pools are started) 
       many threads will enter concurrently 
       we must only initialize this table once */
    if (php_ort_atomic_addref(
            &__ort_math_promotion_schema.gc.refcount) > 1) {
        return;
    }

    zend_hash_init(
        &__ort_math_promotion_schema, 12,
        NULL,
        NULL, /* no destroying these */
        1);   /* immutable */

#define ORT_MATH_PROMOTION_SCHEMA_ADD(symbol); do {   \
    zend_hash_str_add_ptr(                            \
        &__ort_math_promotion_schema,                 \
        ZEND_STRL(#symbol),                           \
        (void*) &ort_math_promotion_schema_##symbol); \
} while (0)
    ORT_MATH_PROMOTION_SCHEMA_ADD(abs);
    ORT_MATH_PROMOTION_SCHEMA_ADD(acos);
    ORT_MATH_PROMOTION_SCHEMA_ADD(add);
    ORT_MATH_PROMOTION_SCHEMA_ADD(arccos);
    ORT_MATH_PROMOTION_SCHEMA_ADD(arccosh);
    ORT_MATH_PROMOTION_SCHEMA_ADD(arcsin);
    ORT_MATH_PROMOTION_SCHEMA_ADD(arcsinh);
    ORT_MATH_PROMOTION_SCHEMA_ADD(arctan);
    ORT_MATH_PROMOTION_SCHEMA_ADD(arctanh);
    ORT_MATH_PROMOTION_SCHEMA_ADD(argmax);
    ORT_MATH_PROMOTION_SCHEMA_ADD(argmin);
    ORT_MATH_PROMOTION_SCHEMA_ADD(asin);
    ORT_MATH_PROMOTION_SCHEMA_ADD(atan);
    ORT_MATH_PROMOTION_SCHEMA_ADD(cos);
    ORT_MATH_PROMOTION_SCHEMA_ADD(cbrt);
    ORT_MATH_PROMOTION_SCHEMA_ADD(ceil);
    ORT_MATH_PROMOTION_SCHEMA_ADD(cosh);
    ORT_MATH_PROMOTION_SCHEMA_ADD(div);
    ORT_MATH_PROMOTION_SCHEMA_ADD(dot);
    ORT_MATH_PROMOTION_SCHEMA_ADD(exp);
    ORT_MATH_PROMOTION_SCHEMA_ADD(exp2);
    ORT_MATH_PROMOTION_SCHEMA_ADD(floor);
    ORT_MATH_PROMOTION_SCHEMA_ADD(log);
    ORT_MATH_PROMOTION_SCHEMA_ADD(log2);
    ORT_MATH_PROMOTION_SCHEMA_ADD(log10);
    ORT_MATH_PROMOTION_SCHEMA_ADD(matmul);
    ORT_MATH_PROMOTION_SCHEMA_ADD(max);
    ORT_MATH_PROMOTION_SCHEMA_ADD(mean);
    ORT_MATH_PROMOTION_SCHEMA_ADD(min);
    ORT_MATH_PROMOTION_SCHEMA_ADD(mod);
    ORT_MATH_PROMOTION_SCHEMA_ADD(mul);
    ORT_MATH_PROMOTION_SCHEMA_ADD(neg);
    ORT_MATH_PROMOTION_SCHEMA_ADD(pow);
    ORT_MATH_PROMOTION_SCHEMA_ADD(recip);
    ORT_MATH_PROMOTION_SCHEMA_ADD(round);
    ORT_MATH_PROMOTION_SCHEMA_ADD(sign);
    ORT_MATH_PROMOTION_SCHEMA_ADD(sin);
    ORT_MATH_PROMOTION_SCHEMA_ADD(sinh);
    ORT_MATH_PROMOTION_SCHEMA_ADD(softmax);
    ORT_MATH_PROMOTION_SCHEMA_ADD(sqrt);
    ORT_MATH_PROMOTION_SCHEMA_ADD(sub);
    ORT_MATH_PROMOTION_SCHEMA_ADD(sum);
    ORT_MATH_PROMOTION_SCHEMA_ADD(tan);
    ORT_MATH_PROMOTION_SCHEMA_ADD(tanh);
    ORT_MATH_PROMOTION_SCHEMA_ADD(trunc);
#undef ORT_MATH_PROMOTION_SCHEMA_ADD
}

ort_math_promotion_schema_t*
    ort_math_promotion_schema_symbol(
        zend_string* name) {
    /* php doesn't care about case */
    zend_string* key = zend_string_tolower(name);
    ort_math_promotion_schema_t* symbol =
        zend_hash_find_ptr(
            &__ort_math_promotion_schema, key);
    zend_string_release(key);
    return symbol;
}

void ort_math_promotion_shutdown(void) {
    if (php_ort_atomic_delref(
            &__ort_math_promotion_schema.gc.refcount) == 0) {
        zend_hash_destroy(&__ort_math_promotion_schema);
    }
}