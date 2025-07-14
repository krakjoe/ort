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

#include "alloc.h"
#include "tensor.h"
#include "status.h"

#ifdef ZTS
static MUTEX_T php_ort_tensor_mutex;
#endif

static HashTable php_ort_tensors;

zend_class_entry *php_ort_tensor_interface_ce;
zend_class_entry *php_ort_tensor_persistent_ce;
zend_class_entry *php_ort_tensor_transient_ce;
zend_object_handlers php_ort_tensor_handlers;

// Recursive shape/data validator
static inline zend_bool php_ort_tensor_validate_next(ONNXTensorElementDataType type, zend_long rank, zend_long *dimensions, zval *node, zend_long depth) {
    if (depth == rank) {
        // Leaf node — must match scalar type
        switch (type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                return Z_TYPE_P(node) == IS_DOUBLE || Z_TYPE_P(node) == IS_LONG;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                return Z_TYPE_P(node) == IS_LONG;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                // UINT64 is not supported due to PHP's signed 64-bit integer limitation
                php_ort_status_flow(!SUCCESS,
                {
                    return 0;
                },
                php_ort_status_tensor_invalidtype_ce,
                "UINT64 tensor type is not supported (values exceed PHP integer range)");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                return Z_TYPE_P(node) == IS_TRUE || Z_TYPE_P(node) == IS_FALSE;
            
                default: php_ort_status_flow(!SUCCESS,
                {
                    return 0;
                },
                php_ort_status_tensor_invalidtype_ce,
                "unknown data type (%zd) provided",
                (zend_long) type);
        }
    }

    if (Z_TYPE_P(node) != IS_ARRAY) {
        return 0;
    }

    HashTable *ht = Z_ARRVAL_P(node);
    if (zend_hash_num_elements(ht) != dimensions[depth]) {
        return 0;
    }

    for (zend_long i = 0; i < dimensions[depth]; i++) {
        zval *child = zend_hash_index_find(ht, i);
        if (!child || !php_ort_tensor_validate_next(
                type, rank, dimensions, child, depth + 1)) {
            return 0;
        }
    }

    return 1;
}

static zend_always_inline zend_bool php_ort_tensor_validate(zval *shape, zend_string *name, zval *data, ONNXTensorElementDataType type) {
    zend_long dimensions[16];
    zend_long rank = 0;
    
    rank = zend_hash_num_elements(Z_ARRVAL_P(shape));
    
    // Handle scalar tensor case (rank 0, empty shape array)
    if (rank == 0) {
        // For scalar tensors, data must be a single-element array
        if (Z_TYPE_P(data) != IS_ARRAY) {
            php_ort_status_flow(!SUCCESS,
            {
                return 0;
            },
            php_ort_status_tensor_invaliddata_ce,
            "scalar tensor data must be provided as a single-element array");
        }
        
        // Check that the data array has exactly one element
        if (zend_hash_num_elements(Z_ARRVAL_P(data)) != 1) {
            php_ort_status_flow(!SUCCESS,
            {
                return 0;
            },
            php_ort_status_tensor_invaliddata_ce,
            "scalar tensor must have exactly one data element");
        }
        
        // Get the first (and only) element
        zval *scalar = zend_hash_index_find(Z_ARRVAL_P(data), 0);
        if (!scalar) {
            return 0;
        }
        
        // Validate the scalar element type
        switch (type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                return Z_TYPE_P(scalar) == IS_DOUBLE || Z_TYPE_P(scalar) == IS_LONG;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                return Z_TYPE_P(scalar) == IS_LONG;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                // UINT64 is not supported due to PHP's signed 64-bit integer limitation
                php_ort_status_flow(!SUCCESS,
                {
                    return 0;
                },
                php_ort_status_tensor_invalidtype_ce,
                "UINT64 tensor type is not supported (values exceed PHP integer range)");
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                return Z_TYPE_P(scalar) == IS_TRUE || Z_TYPE_P(scalar) == IS_FALSE;
            
            default: php_ort_status_flow(!SUCCESS,
            {
                return 0;
            },
            php_ort_status_tensor_invalidtype_ce,
            "unknown data type (%zd) provided",
            (zend_long) type);
        }
    }

    php_ort_status_flow(
        (rank > 16),
        return 0,
        php_ort_status_tensor_invalidshape_ce,
        "invalid shape information (must not exceed 16 dimensions)");

    // Extract and validate dimensions
    zend_long index = 0;
    zval *next;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape), next) {
        php_ort_status_flow(
            (Z_TYPE_P(next) != IS_LONG || Z_LVAL_P(next) <= 0),
            return 0,
            php_ort_status_tensor_invalidshape_ce,
            "shape information must be an array of positive integers");

        php_ort_status_flow(
            (!zend_hash_index_exists(Z_ARRVAL_P(shape), index)),
            return 0,
            php_ort_status_tensor_invalidshape_ce,
            "shape must be a packed array, index %zd is missing",
            index);

        dimensions[index++] = Z_LVAL_P(next);
    } ZEND_HASH_FOREACH_END();

    php_ort_status_flow(
        !php_ort_tensor_validate_next(type, rank, dimensions, data, 0),
        return 0,
        php_ort_status_tensor_invaliddata_ce,
        "validation of data according to the shape provided has failed");

    return 1;
}

static inline zend_bool php_ort_tensor_flatten(ort_tensor_t* tensor, size_t *offset, size_t size, zval *node, size_t depth) {
    // Special case for scalar tensor (0 dimensions)
    if (tensor->dimensions == 0) {
        // For scalar tensors, we expect the data to be a single-element array
        if (Z_TYPE_P(node) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_P(node)) != 1) {
            return 0;
        }
        
        // Get the scalar value (first element of the array)
        zval *scalar = zend_hash_index_find(Z_ARRVAL_P(node), 0);
        if (!scalar) {
            return 0;
        }
        
        void *target = (char *)tensor->data + ((*offset) * size);
        switch (tensor->type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                *(float *)target = Z_TYPE_P(scalar) == IS_DOUBLE
                    ? (float)Z_DVAL_P(scalar)
                    : (float)Z_LVAL_P(scalar);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                *(double *)target = Z_TYPE_P(scalar) == IS_DOUBLE
                    ? Z_DVAL_P(scalar)
                    : (double)Z_LVAL_P(scalar);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                *(int8_t *)target = (int8_t)Z_LVAL_P(scalar);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                *(int16_t *)target = (int16_t)Z_LVAL_P(scalar);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                *(int32_t *)target = (int32_t)Z_LVAL_P(scalar);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                *(int64_t *)target = (int64_t)Z_LVAL_P(scalar);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                *(uint8_t *)target = (uint8_t)Z_LVAL_P(scalar);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                *(uint16_t *)target = (uint16_t)Z_LVAL_P(scalar);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                *(uint32_t *)target = (uint32_t)Z_LVAL_P(scalar);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                // UINT64 is not supported - this should not be reached due to validation
                return 0;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                *(uint8_t *)target = (Z_TYPE_P(scalar) == IS_TRUE) ? 1 : 0;
                break;
        }
        (*offset)++;
        return 1;
    }

    if (depth == tensor->dimensions) {
        void *target = (char *)tensor->data + ((*offset) * size);
        switch (tensor->type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                *(float *)target = Z_TYPE_P(node) == IS_DOUBLE
                    ? (float)Z_DVAL_P(node)
                    : (float)Z_LVAL_P(node);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                *(double *)target = Z_TYPE_P(node) == IS_DOUBLE
                    ? Z_DVAL_P(node)
                    : (double)Z_LVAL_P(node);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                *(int8_t *)target = (int8_t)Z_LVAL_P(node);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                *(int16_t *)target = (int16_t)Z_LVAL_P(node);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                *(int32_t *)target = (int32_t)Z_LVAL_P(node);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                *(int64_t *)target = (int64_t)Z_LVAL_P(node);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                *(uint8_t *)target = (uint8_t)Z_LVAL_P(node);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                *(uint16_t *)target = (uint16_t)Z_LVAL_P(node);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                *(uint32_t *)target = (uint32_t)Z_LVAL_P(node);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                // UINT64 is not supported - this should not be reached due to validation
                return 0;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                *(uint8_t *)target = (Z_TYPE_P(node) == IS_TRUE) ? 1 : 0;
                break;
        }
        (*offset)++;
        return 1;
    }

    for (int64_t i = 0; i < tensor->shape[depth]; i++) {
        zval *child = zend_hash_index_find(Z_ARRVAL_P(node), i);
        if (!php_ort_tensor_flatten(tensor, offset, size, child, depth + 1)) {
            return 0;
        }
    }

    return 1;
}

static zend_always_inline zend_bool php_ort_tensor_allocate_persistent(ort_tensor_t *tensor, zend_string *name, zval *shape, zval *data, ONNXTensorElementDataType type) {
    size_t i = 0, offset = 0;
    size_t size;

    tensor->name       = php_ort_string_copy(name);
    tensor->dimensions = zend_hash_num_elements(Z_ARRVAL_P(shape));
    tensor->type       = type;
    tensor->owner      = PHP_ORT_OWN_HEAP;

    // Handle scalar tensor case (empty shape array)
    if (tensor->dimensions == 0) {
        tensor->shape    = NULL;  // No shape array needed for scalars
        tensor->elements = 1;     // A scalar has exactly one element
        tensor->data     = ort_alloc(php_ort_tensor_sizeof(tensor), 1);
        
        return php_ort_tensor_flatten(tensor, &offset, php_ort_tensor_sizeof(tensor), data, 0);
    }
    
    // Normal case for non-scalar tensors
    tensor->shape    = pemalloc(tensor->dimensions * sizeof(int64_t), 1);
    tensor->elements = 1;

    // Copy shape and compute total number of elements
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape), zval *dim) {
        int64_t size = (int64_t)Z_LVAL_P(dim);
        tensor->shape[i++] = size;
        tensor->elements *= size;
    } ZEND_HASH_FOREACH_END();

    tensor->data = ort_alloc(
        php_ort_tensor_sizeof(tensor), tensor->elements);

    return php_ort_tensor_flatten(tensor, &offset, php_ort_tensor_sizeof(tensor), data, 0);
}

static zend_always_inline zend_bool php_ort_tensor_allocate_transient(ort_tensor_t *tensor, zval *shape, zval *data, ONNXTensorElementDataType type) {
    size_t i = 0, offset = 0;
    size_t size;

    tensor->name       = NULL;
    tensor->dimensions = zend_hash_num_elements(Z_ARRVAL_P(shape));
    tensor->type       = type;
    tensor->owner      = PHP_ORT_OWN_ZEND;

    // Handle scalar tensor case (empty shape array)
    if (tensor->dimensions == 0) {
        tensor->shape    = NULL;  // No shape array needed for scalars
        tensor->elements = 1;     // A scalar has exactly one element
        tensor->data     = ort_alloc(1, php_ort_tensor_sizeof(tensor));
        
        return php_ort_tensor_flatten(tensor, &offset, php_ort_tensor_sizeof(tensor), data, 0);
    }
    
    // Normal case for non-scalar tensors
    tensor->shape    = ecalloc(tensor->dimensions, sizeof(int64_t));
    tensor->elements = 1;

    // Copy shape and compute total number of elements
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(shape), zval *dim) {
        int64_t size = (int64_t)Z_LVAL_P(dim);
        tensor->shape[i++] = size;
        tensor->elements *= size;
    } ZEND_HASH_FOREACH_END();

    tensor->data = ort_alloc(
        php_ort_tensor_sizeof(tensor), tensor->elements);

    return php_ort_tensor_flatten(tensor, &offset, php_ort_tensor_sizeof(tensor), data, 0);
}

static void ort_tensor_free(ort_tensor_t *tensor) {
    zend_bool persistent = 
        (tensor->owner == PHP_ORT_OWN_HEAP) ? 1 : 0;

    if (tensor->shape) {
        pefree(tensor->shape, persistent);
    }

    if (!tensor->parent && tensor->data && !tensor->value) {
        ort_free(tensor->data);
    }

    if (tensor->value) {
        api->ReleaseValue(tensor->value);
    }

    if (tensor->name && !persistent) {
        zend_string_free(tensor->name);
    }

    if (tensor->parent) {
        ort_tensor_release(tensor->parent);
    }

    pefree(tensor, persistent);
}

void ort_tensor_release(ort_tensor_t *tensor) {
    if (!tensor) {
        return;
    }

    if (php_ort_atomic_delref(&tensor->refcount) == 0){
        ort_tensor_free(tensor);
    }
}

static void php_ort_tensor_del(zval *zv) {
    ort_tensor_release(
        ((ort_tensor_t*)
            Z_PTR_P(zv)));
}

static zend_bool php_ort_tensor_construct_persistent(ort_tensor_t *tensor, zend_string *name, zval *shape, zval *data, ONNXTensorElementDataType type){
    if (!php_ort_tensor_validate(shape, name, data, type)) {
        return 0;
    }

    if (!php_ort_tensor_allocate_persistent(tensor, name, shape, data, type)) {
        return 0;
    }

    return 1;
}

static zend_bool php_ort_tensor_construct_transient(ort_tensor_t *tensor, zval *shape, zval *data, ONNXTensorElementDataType type){
    if (!php_ort_tensor_validate(shape, NULL, data, type)) {
        return 0;
    }

    if (!php_ort_tensor_allocate_transient(tensor, shape, data, type)) {
        return 0;
    }

    return 1;
}

OrtValue* php_ort_tensor_value(php_ort_tensor_t* ort) {
    OrtMemoryInfo* mi;
    OrtValue* value = NULL;
    OrtStatus* status;

    php_ort_status_flow(
        (status = api->CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &mi)),
        {
            api->ReleaseStatus(status);

            return NULL;
        },
        php_ort_status_tensor_invalidmemory_ce,
        "failed to allocate MemoryInfo* for Tensor conversion: %s",
        api->GetErrorMessage(status));

    php_ort_status_flow(
        (status = api->CreateTensorWithDataAsOrtValue(
            mi,
            ort->object->data,
            ort->object->elements * php_ort_tensor_sizeof(ort->object),
            ort->object->shape,
            ort->object->dimensions,
            ort->object->type,
            &value)),
        {
            api->ReleaseStatus(status);

            return NULL;
        },
        php_ort_status_tensor_invalidmemory_ce,
        "failed to allocate OrtValue* for Tensor conversion: %s",
        api->GetErrorMessage(status));

    api->ReleaseMemoryInfo(mi);

    return value;
}

ort_tensor_t* php_ort_tensor_object(OrtValue* value) {
    ort_tensor_t  *tensor = pecalloc(1, sizeof(ort_tensor_t), 0);

    tensor->refcount = 1;
    tensor->owner    = PHP_ORT_OWN_ZEND;
    tensor->value    = value;

    OrtTensorTypeAndShapeInfo* otsi;

    php_ort_status_flow(
        api->GetTensorTypeAndShape(value, &otsi), {
            ort_tensor_free(tensor);

            return NULL;
        },
        php_ort_status_tensor_invalidshape_ce,
        "failed to determine shape for OrtValue* conversion");

    php_ort_status_flow(
        api->GetTensorElementType(otsi, &tensor->type),
        {
            ort_tensor_free(tensor);

            return NULL;
        },
        php_ort_status_tensor_invalidshape_ce,
        "failed to determine type for OrtValue* conversion");

    php_ort_status_flow(
        api->GetDimensionsCount(otsi, &tensor->dimensions),
        {
            ort_tensor_free(tensor);

            return NULL;
        },
        php_ort_status_tensor_invalidshape_ce,
        "failed to determine dimension count for OrtValue* conversion");

    // For non-scalar tensors, allocate shape array and populate it
    if (tensor->dimensions > 0) {
        tensor->shape = pecalloc(tensor->dimensions, sizeof(int64_t), 0);

        php_ort_status_flow(
            api->GetDimensions(otsi, tensor->shape, tensor->dimensions),
            {
                ort_tensor_free(tensor);

                return NULL;
            },
            php_ort_status_tensor_invalidshape_ce,
            "failed to fetch dimensions for OrtValue* conversion");
    } else {
        // For scalar tensors, shape remains NULL
        tensor->shape = NULL;
    }

    php_ort_status_flow(
        api->GetTensorShapeElementCount(otsi, &tensor->elements),
        {
            ort_tensor_free(tensor);

            return NULL;
        },
        php_ort_status_tensor_invalidshape_ce,
        "failed to determine element count for OrtValue* conversion");

    php_ort_status_flow(
        api->GetTensorMutableData(value, &tensor->data),
        {
            ort_tensor_free(tensor);

            return NULL;
        },
        php_ort_status_tensor_invaliddata_ce,
        "failed to fetch data for OrtValue* conversion");

    api->ReleaseTensorTypeAndShapeInfo(otsi);

    return tensor;
}

static zend_always_inline size_t php_ort_tensor_indexof(ort_tensor_t *tensor, int64_t *coords) {
    // For scalar tensors, always return index 0
    if (tensor->dimensions == 0) {
        return 0;
    }
    
    size_t index = 0;
    size_t stride = 1;

    for (int64_t i = tensor->dimensions - 1; i >= 0; i--) {
        index += 
            coords[i] * stride;
        stride *= tensor->shape[i];
    }

    return index;
}

ZEND_BEGIN_ARG_INFO_EX(php_ort_tensor_persistent_construct_arginfo, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, type, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Tensor_Persistent, __construct)
{
    php_ort_tensor_t *ort = php_ort_tensor_fetch(Z_OBJ(EX(This)));

    zend_string *name;
    zval        *shape = NULL;
    zval        *data  = NULL;
    zend_long    type  = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;

    ZEND_PARSE_PARAMETERS_START(1, 4);
        Z_PARAM_STR(name)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY(shape)
        Z_PARAM_ARRAY(data)
        Z_PARAM_LONG(type)
    ZEND_PARSE_PARAMETERS_END();

#ifdef ZTS
    php_ort_status_flow(
        tsrm_mutex_lock(php_ort_tensor_mutex) != SUCCESS,
        return,
        php_ort_status_safetyerror_ce,
        "it was not possible to acquire the tensor mutex, something is terribly wrong");
#endif

    if (!(ort->object = zend_hash_find_ptr(&php_ort_tensors, name))) {
#ifdef ZTS
        php_ort_status_flow(
            (!shape || !data),
            {
                php_ort_status_flow(
                    tsrm_mutex_unlock(php_ort_tensor_mutex) != SUCCESS,
                    return,
                    php_ort_status_safetyerror_ce,
                    "it was not possible to release the tensor mutex, something is terribly wrong");
                return;
            },
            php_ort_status_tensor_notfound_ce, 
            "Could not find the Tensor named \"%s\"",
                ZSTR_VAL(name));
#else
        php_ort_status_flow(
            (!shape || !data),
            {
                return;
            },
            php_ort_status_tensor_notfound_ce, 
            "Could not find the Tensor named \"%s\"",
                ZSTR_VAL(name));
#endif

        ort_tensor_t *tensor = 
            pecalloc(1, sizeof(ort_tensor_t), 1);

        tensor->refcount = 1;

        if (!php_ort_tensor_construct_persistent(tensor, name, shape, data, type)) {
#ifdef ZTS
            php_ort_status_flow(
                tsrm_mutex_unlock(php_ort_tensor_mutex) != SUCCESS,
                return,
                php_ort_status_safetyerror_ce,
                "it was not possible to release the tensor mutex, something is terribly wrong");
#endif
            return;
        }

        ort->object = zend_hash_add_ptr(
            &php_ort_tensors,
            tensor->name,
            tensor);

        ort->object->refcount++;
    } else {
        php_ort_atomic_addref(&ort->object->refcount);
    }

#ifdef ZTS
    php_ort_status_flow(
        tsrm_mutex_unlock(php_ort_tensor_mutex) != SUCCESS,
        return,
        php_ort_status_safetyerror_ce,
        "it was not possible to release the tensor mutex, something is terribly wrong");
#endif
}

ZEND_BEGIN_ARG_INFO_EX(php_ort_tensor_transient_construct_arginfo, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, type, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Tensor_Transient, __construct)
{
    php_ort_tensor_t *ort = php_ort_tensor_fetch(Z_OBJ(EX(This)));

    zval        *shape = NULL;
    zval        *data  = NULL;
    zend_long    type  = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;

    ZEND_PARSE_PARAMETERS_START(2, 3);
        Z_PARAM_ARRAY(shape)
        Z_PARAM_ARRAY(data)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(type)
    ZEND_PARSE_PARAMETERS_END();

    ort_tensor_t *tensor = ecalloc(1, sizeof(ort_tensor_t));
    tensor->refcount = 1;

    if (!php_ort_tensor_construct_transient(tensor, shape, data, type)) {
        efree(tensor);
        return;
    }

    ort->object = tensor;
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_tensor_isPersistent_arginfo, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Tensor, isPersistent)
{
    php_ort_tensor_t* ort =
        php_ort_tensor_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    RETURN_BOOL(ort->object->owner == PHP_ORT_OWN_HEAP);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_tensor_getName_arginfo, 0, 0, IS_STRING, 1)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Tensor, getName)
{
    php_ort_tensor_t* ort =
        php_ort_tensor_fetch(Z_OBJ(EX(This)));
    
    ZEND_PARSE_PARAMETERS_NONE();

    if (!ort->object->name) {
        return;
    }

    RETURN_STR_COPY(ort->object->name);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_tensor_getType_arginfo, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Tensor, getType)
{
    php_ort_tensor_t* ort =
        php_ort_tensor_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    RETURN_LONG(ort->object->type);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_tensor_getTypeName_arginfo, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Tensor, getTypeName)
{
    php_ort_tensor_t* ort =
        php_ort_tensor_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    /* @todo(krakjoe) this performs terribly ... */
    RETURN_STRING(php_ort_type_name(ort->object->type));
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_tensor_getShape_arginfo, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Tensor, getShape)
{
    php_ort_tensor_t* ort =
        php_ort_tensor_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    array_init(return_value);
    for (int64_t dimension = 0; dimension < ort->object->dimensions; dimension++) {
        add_next_index_long(
            return_value, ort->object->shape[dimension]);
    }
}

static zend_always_inline ort_tensor_t* php_ort_tensor_transpose(ort_tensor_t* input, zval *axis, zval* return_value) {
    int64_t ndim = input->dimensions;
    int64_t* perm = NULL;
    int64_t* inv_perm = NULL;
    int64_t* out_shape = NULL;
    int64_t* in_strides = NULL;
    int64_t* out_strides = NULL;
    size_t type_size = php_ort_tensor_sizeof(input);
    size_t numel = input->elements;
    ort_tensor_t* result = NULL;

    // Argument validation and axes parsing
    if (ndim == 0) {
        // Scalar: transpose is a no-op, just return a copy
        result = pecalloc(1, sizeof(ort_tensor_t), 0);
        result->refcount = 1;
        result->owner = PHP_ORT_OWN_ZEND;
        result->type = input->type;
        result->dimensions = 0;
        result->elements = 1;
        result->shape = NULL;
        result->data = ort_alloc(type_size, 1);
        ort_memcpy(result->data, input->data, type_size);
        goto __php_ort_tensor_transpose_done;
    }

    perm = ecalloc(ndim, sizeof(int64_t));
    out_shape = ecalloc(ndim, sizeof(int64_t));
    in_strides = ecalloc(ndim, sizeof(int64_t));
    out_strides = ecalloc(ndim, sizeof(int64_t));
    inv_perm = ecalloc(ndim, sizeof(int64_t));

    // Parse axes argument
    if (axis) {
        php_ort_status_flow(
            (zend_hash_num_elements(Z_ARRVAL_P(axis)) != ndim),
            {
                goto __php_ort_tensor_transpose_failed;
            },
            php_ort_status_tensor_invalidshape_ce,
            "axes array must be an array of length %zd", ndim);

        // Fill perm from user axes, handle negatives, check for duplicates
        zend_bool* seen = ecalloc(ndim, sizeof(zend_bool));
        int64_t i = 0;
        zval* zv;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(axis), zv) {
            php_ort_status_flow(
                (Z_TYPE_P(zv) != IS_LONG),
                {
                    efree(seen);
                    goto __php_ort_tensor_transpose_failed;
                },
                php_ort_status_tensor_invalidshape_ce,
                "axes array must contain only integers");
            
            int64_t ax = Z_LVAL_P(zv);
            if (ax < 0)
                ax += ndim;
            
            php_ort_status_flow(
                (ax < 0 || ax >= ndim),
                {
                    efree(seen);
                    goto __php_ort_tensor_transpose_failed;
                },
                php_ort_status_tensor_invalidshape_ce,
                "axis value %zd out of range [0, %zd)", ax, ndim);

            php_ort_status_flow(
                (seen[ax]),
                {
                    efree(seen);
                    goto __php_ort_tensor_transpose_failed;
                },
                php_ort_status_tensor_invalidshape_ce,
                "duplicate axis value %zd in axes", ax);

            seen[ax] = 1;
            perm[i++] = ax;
        } ZEND_HASH_FOREACH_END();
        efree(seen);
    } else {
        // Default: reverse axes
        for (int64_t i = 0; i < ndim; i++) {
            perm[i] = ndim - 1 - i;
        }
    }

    // Compute output shape and strides
    for (int64_t i = 0; i < ndim; i++) {
        out_shape[i] = input->shape[perm[i]];
    }
    // Input strides
    in_strides[ndim-1] = 1;
    for (int64_t i = ndim-2; i >= 0; i--) {
        in_strides[i] = in_strides[i+1] * input->shape[i+1];
    }
    // Output strides
    out_strides[ndim-1] = 1;
    for (int64_t i = ndim-2; i >= 0; i--) {
        out_strides[i] = out_strides[i+1] * out_shape[i+1];
    }
    // Inverse permutation: inv_perm[perm[i]] = i
    for (int64_t i = 0; i < ndim; i++) {
        inv_perm[perm[i]] = i;
    }

    // Allocate result tensor
    result = pecalloc(1, sizeof(ort_tensor_t), 0);
    result->refcount = 1;
    result->owner = PHP_ORT_OWN_ZEND;
    result->type = input->type;
    result->dimensions = ndim;
    result->elements = numel;
    result->shape = pecalloc(ndim, sizeof(int64_t), 0);
    memcpy(result->shape,
        out_shape, ndim * sizeof(int64_t));
    result->data = ort_alloc(type_size, numel);

    // Main permutation loop
    int64_t* in_coords =
        ecalloc(ndim, sizeof(int64_t));
    int64_t* out_coords =
        ecalloc(ndim, sizeof(int64_t));
    for (size_t idx = 0; idx < numel; idx++) {
        // Compute input coordinates from flat idx
        size_t rem = idx;
        for (int64_t i = 0; i < ndim; i++) {
            in_coords[i] = rem / in_strides[i];
            rem = rem % in_strides[i];
        }
        // Permute coordinates
        for (int64_t i = 0; i < ndim; i++) {
            out_coords[i] = in_coords[perm[i]];
        }
        // Compute output flat index
        size_t out_idx = 0;
        for (int64_t i = 0; i < ndim; i++) {
            out_idx += out_coords[i] * out_strides[i];
        }
        // Copy element
        ort_memcpy(
            (char*)result->data + out_idx * type_size,
            (char*)input->data + idx * type_size,
            type_size);
    }
    efree(in_coords);
    efree(out_coords);

__php_ort_tensor_transpose_done:
    object_init_ex(return_value,
        php_ort_tensor_transient_ce);
    php_ort_tensor_t* rv =
        php_ort_tensor_fetch(
            Z_OBJ_P(return_value));
    rv->object = result;

    if (perm)
        efree(perm);
    if (out_shape)
        efree(out_shape);
    if (in_strides)
        efree(in_strides);
    if (out_strides)
        efree(out_strides);
    if (inv_perm)
        efree(inv_perm);
    return result;

__php_ort_tensor_transpose_failed:
    if (perm)
        efree(perm);
    if (out_shape)
        efree(out_shape);
    if (in_strides)
        efree(in_strides);
    if (out_strides)
        efree(out_strides);
    if (inv_perm)
        efree(inv_perm);
    return NULL;
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_tensor_transpose_arginfo, 0, 0, ONNX\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, axis, IS_ARRAY, 1)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Tensor, transpose)
{
    php_ort_tensor_t* ort =
        php_ort_tensor_fetch(Z_OBJ(EX(This)));
    zval* axis = NULL;

    ZEND_PARSE_PARAMETERS_START(0, 1);
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(axis)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_transpose(ort->object, axis, return_value);
}

static zend_always_inline ort_tensor_t* php_ort_tensor_slice(ort_tensor_t* input, zval* start, zval* end, zval* axis, zval* return_value) {
    // Handle axis parameter - if provided, validate it
    zend_bool has_axis = (axis != NULL);
    HashTable *axis_ht = has_axis ? Z_ARRVAL_P(axis) : NULL;

    if (has_axis) {
        // When axis is provided, start and end should match axis array length
        php_ort_status_flow(
            (zend_hash_num_elements(Z_ARRVAL_P(start)) != zend_hash_num_elements(axis_ht) ||
             zend_hash_num_elements(Z_ARRVAL_P(end))   != zend_hash_num_elements(axis_ht)), 
            return NULL,
            php_ort_status_tensor_invalidshape_ce,
            "when axis is provided, start and end arrays must have same length as axis array (%zd)",
            zend_hash_num_elements(axis_ht));
    } else {
        // When no axis provided, start and end must match tensor dimensions
        php_ort_status_flow(
            (zend_hash_num_elements(Z_ARRVAL_P(start)) != input->dimensions ||
             zend_hash_num_elements(Z_ARRVAL_P(end))   != input->dimensions), 
            return NULL,
            php_ort_status_tensor_invalidshape_ce,
            "start and end arrays must have same length as tensor dimensions (%zd)",
            input->dimensions);
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);

    php_ort_tensor_t* ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    
    // Allocate new tensor structure
    ort->object = pecalloc(1, sizeof(ort_tensor_t), 0);
    
    // Set up parent relationship
    ort->object->parent = input;
    php_ort_atomic_addref(
        &ort->object->parent->refcount);

    ort->object->refcount = 1;
    ort->object->owner = PHP_ORT_OWN_ZEND;
    ort->object->type = input->type;

    int64_t *starting = pecalloc(input->dimensions, sizeof(int64_t), 0);
    int64_t *ending = pecalloc(input->dimensions, sizeof(int64_t), 0);
    int64_t *slicing = pecalloc(input->dimensions, sizeof(int64_t), 0);

    // Initialize with full range for all dimensions
    for (int64_t idim = 0; idim < input->dimensions; idim++) {
        starting[idim] = 0;
        ending[idim]   = input->shape[idim];
        slicing[idim]  = input->shape[idim];
    }

    // Process based on whether axis is provided
    if (has_axis) {
        // Process each axis specification
        zval *zaxis, *zstart, *zend;
        zend_ulong iaxis = 0;
        
        ZEND_HASH_FOREACH_VAL(axis_ht, zaxis) {
            php_ort_status_flow(
                (Z_TYPE_P(zaxis) != IS_LONG),
                {
                    pefree(starting, 0);
                    pefree(ending, 0);
                    pefree(slicing, 0);
                    return NULL;
                },
                php_ort_status_tensor_invalidshape_ce,
                "axis array must contain integers");
            
            int64_t idim = Z_LVAL_P(zaxis);

            php_ort_status_flow(
                (idim < 0 || idim >= input->dimensions),
                {
                    pefree(starting, 0);
                    pefree(ending, 0);
                    pefree(slicing, 0);

                    return NULL;
                },
                php_ort_status_tensor_invalidshape_ce,
                "axis value %zd out of range [0, %zd)", idim, input->dimensions);
            
            // Get corresponding start and end values using the same index
            zstart = zend_hash_index_find(Z_ARRVAL_P(start), iaxis);
            zend   = zend_hash_index_find(Z_ARRVAL_P(end),   iaxis);
            
            php_ort_status_flow(
                (!zstart || !zend || 
                  Z_TYPE_P(zstart) != IS_LONG || 
                  Z_TYPE_P(zend)   != IS_LONG),
                {
                    pefree(starting, 0);
                    pefree(ending, 0);
                    pefree(slicing, 0);

                    return NULL;
                },
                php_ort_status_tensor_invalidshape_ce,
                "start and end values must be integers");

            int64_t istart = Z_LVAL_P(zstart);
            int64_t iend = Z_LVAL_P(zend);
            
            // Handle negative indices
            if (istart < 0) istart += input->shape[idim];
            if (iend < 0)   iend += input->shape[idim];

            // Validate bounds
            php_ort_status_flow(
                (istart < 0 || istart >= input->shape[idim] ||
                 iend < 0   || iend    > input->shape[idim] ||
                 istart >= iend),
                {
                    pefree(starting, 0);
                    pefree(ending, 0);
                    pefree(slicing, 0);
                    return NULL;
                },
                php_ort_status_tensor_invalidshape_ce,
                "invalid slice bounds for axis %zd: start=%zd, end=%zd, shape=%zd",
                idim, istart, iend, input->shape[idim]);

            starting[idim] = istart;
            ending[idim]   = iend;
            slicing[idim]  = iend - istart;

            iaxis++;
        } ZEND_HASH_FOREACH_END();
    } else {
        // No axis specified, use start and end arrays directly for all dimensions
        for (int64_t idim = 0; idim < input->dimensions; idim++) {
            zval *zstart = zend_hash_index_find(Z_ARRVAL_P(start), idim);
            zval *zend   = zend_hash_index_find(Z_ARRVAL_P(end),   idim);

            php_ort_status_flow(
                (!zstart || !zend ||
                 Z_TYPE_P(zstart) != IS_LONG ||
                 Z_TYPE_P(zend)   != IS_LONG),
                {
                    pefree(starting, 0);
                    pefree(ending, 0);
                    pefree(slicing, 0);
                    return NULL;
                },
                php_ort_status_tensor_invalidshape_ce,
                "start and end values must be integers");

            int64_t istart = Z_LVAL_P(zstart);
            int64_t iend = Z_LVAL_P(zend);

            // Handle negative indices
            if (istart < 0) istart += input->shape[idim];
            if (iend < 0)   iend   += input->shape[idim];

            php_ort_status_flow(
                (istart < 0 || istart >= input->shape[idim] ||
                 iend < 0   || iend   >  input->shape[idim] ||
                 istart     >= iend),
                {
                    pefree(starting, 0);
                    pefree(ending, 0);
                    pefree(slicing, 0);

                    return NULL;
                },
                php_ort_status_tensor_invalidshape_ce,
                "invalid slice bounds for dimension %zd: start=%zd, end=%zd, shape=%zd",
                idim, istart, iend, input->shape[idim]);

            starting[idim] = istart;
            ending[idim]   = iend;
            slicing[idim]  = iend - istart;
        }
    }

    // Set up slice tensor metadata
    ort->object->dimensions = input->dimensions;
    ort->object->shape      = slicing; // Transfer ownership

    // Calculate number of elements in slice
    ort->object->elements = 1;
    for (int64_t idim = 0; idim < ort->object->dimensions; idim++) {
        ort->object->elements *= ort->object->shape[idim];
    }

    // Set data pointer to offset into parent's data (readonly view)
    ort->object->data = (char*)input->data + 
        (php_ort_tensor_indexof(input, starting) * 
            php_ort_tensor_sizeof(input));

    pefree(starting, 0);
    pefree(ending, 0);

    return ort->object;
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_tensor_getSlice_arginfo, 0, 2, ONNX\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, start, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, end,   IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, axis,  IS_ARRAY, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Tensor, getSlice)
{
    php_ort_tensor_t* ort =
        php_ort_tensor_fetch(Z_OBJ(EX(This)));
    zval *start, *end, *axis = NULL;

    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ARRAY(start)
        Z_PARAM_ARRAY(end)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY(axis)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_slice(ort->object, start, end, axis, return_value);
}

static inline zend_bool php_ort_tensor_data(ort_tensor_t *tensor, size_t *offset, size_t size, zval *node, size_t depth) {
    // Special case for scalar tensors (0 dimensions)
    if (tensor->dimensions == 0) {
        // This should not be called for scalar tensors
        // The getData method directly handles scalar case
        php_ort_status_flow(
            !SUCCESS,
            return 0,
            php_ort_status_tensor_invalidshape_ce,
            "php_ort_tensor_data should not be called for scalar tensors");
    }

    // Validate depth bounds
    php_ort_status_flow(
        (depth >= tensor->dimensions),
        return 0,
        php_ort_status_tensor_invalidshape_ce,
        "depth %zd exceeds tensor dimensions %zd", depth, tensor->dimensions);

    // Validate shape at current depth
    php_ort_status_flow(
        (tensor->shape[depth] <= 0),
        return 0,
        php_ort_status_tensor_invalidshape_ce,
        "invalid shape at dimension %zd: %zd", depth, tensor->shape[depth]);

    // Calculate remaining elements from current offset
    size_t remaining_elements = (*offset < tensor->elements) ? (tensor->elements - *offset) : 0;
    
    // Calculate how many elements we can actually extract at this depth
    int64_t elements_to_extract = tensor->shape[depth];
    if (depth == tensor->dimensions - 1) {
        // At leaf level, limit to remaining elements
        elements_to_extract = (remaining_elements < (size_t)tensor->shape[depth]) ? 
                             (int64_t)remaining_elements : tensor->shape[depth];
    }

    // Initialize array with actual size we'll extract
    array_init_size(node, elements_to_extract);

    if (depth == tensor->dimensions - 1) {
        // Leaf level - extract scalar values
        for (int64_t i = 0; i < elements_to_extract; i++) {
            zval val;
            
            // Validate offset bounds before accessing data
            php_ort_status_flow(
                (*offset >= tensor->elements),
                return 0,
                php_ort_status_tensor_invaliddata_ce,
                "data offset %zd exceeds tensor element count %zd", *offset, tensor->elements);

            void *source = (char *)tensor->data + ((*offset) * size);

            switch (tensor->type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    ZVAL_DOUBLE(&val, *(float *)source);
                    break;

                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                    ZVAL_DOUBLE(&val, *(double *)source);
                    break;

                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                    ZVAL_LONG(&val, *(int8_t *)source);
                    break;

                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                    ZVAL_LONG(&val, *(int16_t *)source);
                    break;

                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                    ZVAL_LONG(&val, *(int32_t *)source);
                    break;

                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                    ZVAL_LONG(&val, *(int64_t *)source);
                    break;

                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                    ZVAL_LONG(&val, *(uint8_t *)source);
                    break;

                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                    ZVAL_LONG(&val, *(uint16_t *)source);
                    break;

                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                    ZVAL_LONG(&val, *(uint32_t *)source);
                    break;

                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                    // UINT64 is not supported - this should not be reached
                    php_ort_status_flow(
                        !SUCCESS,
                        {
                            return 0;
                        },
                        php_ort_status_tensor_invalidtype_ce,
                        "UINT64 tensor type is not supported (values exceed PHP integer range)");

                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                    ZVAL_BOOL(&val, *(uint8_t *)source ? 1 : 0);
                    break;

                default:
                    php_ort_status_flow(
                        !SUCCESS,
                        {
                            return 0;
                        },
                        php_ort_status_tensor_invaliddata_ce,
                        "unsupported tensor data type: %d", tensor->type);
            }

            php_ort_status_flow(
                (zend_hash_index_update(Z_ARRVAL_P(node), i, &val) == NULL),
                return 0,
                php_ort_status_tensor_invaliddata_ce,
                "failed to update array at index %zd", i);

            (*offset)++;
        }
    } else {
        // Recursive case - process sub-dimensions
        // For non-leaf levels, we need to check if we have enough remaining elements
        // to fill the expected sub-structures
        for (int64_t i = 0; i < elements_to_extract; i++) {
            // Check if we still have elements remaining before processing
            if (*offset >= tensor->elements) {
                break; // No more elements available
            }
            
            zval child;
            
            php_ort_status_flow(
                (!php_ort_tensor_data(tensor, offset, size, &child, depth + 1)),
                return 0,
                php_ort_status_tensor_invaliddata_ce,
                "failed to extract data at dimension %zd, index %zd", depth, i);

            php_ort_status_flow(
                (zend_hash_index_update(Z_ARRVAL_P(node), i, &child) == NULL),
                return 0,
                php_ort_status_tensor_invaliddata_ce,
                "failed to update array at dimension %zd, index %zd", depth, i);
        }
    }

    return 1;
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_tensor_getData_arginfo, 0, 0, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, offset, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, depth,  IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Tensor, getData)
{
    php_ort_tensor_t* ort =
        php_ort_tensor_fetch(Z_OBJ(EX(This)));
    zend_long offset = 0, depth = 0;

    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(offset)
        Z_PARAM_LONG(depth)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_status_flow(
        (offset < 0 || offset > ort->object->elements),
        return,
        php_ort_status_tensor_invaliddata_ce,
        "offset %zd out of range [0, %zd]",
        offset, ort->object->elements);

    // Special case for scalar tensors (0 dimensions)
    if (ort->object->dimensions == 0) {
        // For scalar tensors, check if depth parameter was provided
        php_ort_status_flow(
            (ZEND_NUM_ARGS() > 1 && depth != 0),
            return,
            php_ort_status_tensor_invalidshape_ce,
            "depth parameter cannot be used with scalar tensors");

        // Handle offset parameter - for scalar tensors, offset 0 returns the value, any other offset returns empty array
        if (offset > 0) {
            array_init(return_value);
            return;
        }
        
        // For scalar tensors, return a single-element array with the scalar value
        array_init_size(return_value, 1);
        zval val;
        
        void *source = ort->object->data;
        
        switch (ort->object->type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                ZVAL_DOUBLE(&val, *(float *)source);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                ZVAL_DOUBLE(&val, *(double *)source);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                ZVAL_LONG(&val, *(int8_t *)source);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                ZVAL_LONG(&val, *(int16_t *)source);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                ZVAL_LONG(&val, *(int32_t *)source);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                ZVAL_LONG(&val, *(int64_t *)source);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                ZVAL_LONG(&val, *(uint8_t *)source);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                ZVAL_LONG(&val, *(uint16_t *)source);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                ZVAL_LONG(&val, *(uint32_t *)source);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                // UINT64 is not supported - this should not be reached
                php_ort_status_flow(
                    !SUCCESS,
                    return,
                    php_ort_status_tensor_invalidtype_ce,
                    "UINT64 tensor type is not supported (values exceed PHP integer range)");

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                ZVAL_BOOL(&val, *(uint8_t *)source ? 1 : 0);
                break;

            default:
                php_ort_status_flow(
                    !SUCCESS,
                    return,
                    php_ort_status_tensor_invaliddata_ce,
                    "unsupported tensor data type: %d", ort->object->type);
        }
        
        add_index_zval(return_value, 0, &val);
        return;
    }

    // Normal case for non-scalar tensors
    php_ort_status_flow(
        (depth < 0 || depth >= ort->object->dimensions),
        return,
        php_ort_status_tensor_invalidshape_ce,
        "depth %zd out of range [0, %zd)",
        depth, ort->object->dimensions);

    php_ort_status_flow(
        (!php_ort_tensor_data(
            ort->object, 
            &offset,
            php_ort_tensor_sizeof(ort->object), 
            return_value, 
            (size_t)depth)),
        return,
        php_ort_status_tensor_invaliddata_ce,
        "failed to extract tensor data starting at offset %zd, depth %zd", offset, depth);
}

// Helper: Recursively infer shape from a PHP array
// Robust recursive shape inference with raggedness and type checks
static zend_bool php_ort_infer_shape(zval *data, size_t *shape, size_t max, size_t *dimensions) {
    size_t dimension = 0;
    zval *level = data;
    while (Z_TYPE_P(level) == IS_ARRAY) {
        php_ort_status_flow(
            (dimension >= max),
            return 0,
            php_ort_status_tensor_invaliddata_ce,
            "shape exceeds maximum allowed dimensions (%zd)", max);

        size_t len = zend_hash_num_elements(Z_ARRVAL_P(level));

        php_ort_status_flow(
            len == 0,
            return 0,
            php_ort_status_tensor_invaliddata_ce,
            "empty array encountered at dimension %zd (ragged or empty tensor)",
            dimension);

        shape[dimension] = len;

        // Check all elements at this level are arrays or all are scalars
        zend_bool found_array = 0, found_scalar = 0;
        zval *first = NULL;
        zend_ulong idx = 0;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(level), first) { 
            break; 
        } ZEND_HASH_FOREACH_END();
        
        if (first && Z_TYPE_P(first) == IS_ARRAY) {
            found_array = 1;
        } else {
            found_scalar = 1;
        }
        
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(level), zval *sub) {
            if (Z_TYPE_P(sub) == IS_ARRAY) {
                found_array = 1;
                php_ort_status_flow(
                    (zend_hash_num_elements(Z_ARRVAL_P(sub)) !=
                        zend_hash_num_elements(Z_ARRVAL_P(first))),
                    return 0,
                    php_ort_status_tensor_invaliddata_ce,
                    "ragged array: sub-array at dimension %zd has length %zd, expected %zd",
                    dimension+1,
                    zend_hash_num_elements(Z_ARRVAL_P(sub)),
                    zend_hash_num_elements(Z_ARRVAL_P(first)));
            } else {
                found_scalar = 1;
            }

            php_ort_status_flow(
                (Z_TYPE_P(sub) != IS_ARRAY &&
                Z_TYPE_P(sub) != IS_LONG &&
                Z_TYPE_P(sub) != IS_DOUBLE &&
                !(Z_TYPE_P(sub) == IS_TRUE || Z_TYPE_P(sub) == IS_FALSE)),
                return 0,
                php_ort_status_tensor_invaliddata_ce,
                "unsupported type at dimension %zd: %s",
                dimension+1,
                zend_zval_type_name(sub));

            php_ort_status_flow(
                (found_array && found_scalar),
                return 0,
                php_ort_status_tensor_invaliddata_ce,
                "mixed array/scalar types at dimension %zd (ragged tensor)",
                dimension+1);
        } ZEND_HASH_FOREACH_END();
        if (found_array) {
            // Go one level deeper
            level = first;
            dimension++;
        } else {
            // All scalars at this level, this is the last dimension
            dimension++;
            break;
        }
    }
    *dimensions = dimension;
    return 1;
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_tensor_from_arginfo, 0, 2, ONNX\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, type, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Tensor, from)
{
    zval *data;
    zend_long type;
    size_t shape[32];
    size_t dimensions = 0;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ARRAY(data)
        Z_PARAM_LONG(type)
    ZEND_PARSE_PARAMETERS_END();

    if (!php_ort_infer_shape(data, shape, 32, &dimensions)) {
        return;
    }

    php_ort_status_flow(
        (dimensions == 0),
        return,
        php_ort_status_tensor_invaliddata_ce,
        "empty tensor data provided (no dimensions inferred)");

    zval param;
    array_init_size(&param, dimensions);
    for (size_t i = 0; i < dimensions; i++) {
        add_next_index_long(&param, shape[i]);
    }

    zend_class_entry *scope =
        zend_get_executed_scope();
    object_init_ex(return_value, scope);

    zval retval;
    zval params[3];
    ZVAL_ARR(&params[0], Z_ARRVAL(param));
    ZVAL_ARR(&params[1], Z_ARRVAL_P(data));
    ZVAL_LONG(&params[2], type);

    zval constructor;
    ZVAL_STRING(&constructor, "__construct");
    zval result;
    if (SUCCESS == call_user_function(
            EG(function_table),
            return_value,
            &constructor,
            &result,
            3,
            params)) {
        zval_ptr_dtor(&result);
    } else {
        zval_ptr_dtor(return_value);
    }

    zval_ptr_dtor(&constructor);
    zval_ptr_dtor(&param);
}

zend_function_entry php_ort_tensor_interface_methods[] = {
    PHP_ABSTRACT_ME(ONNX_Tensor, isPersistent, php_ort_tensor_isPersistent_arginfo)
    PHP_ABSTRACT_ME(ONNX_Tensor, getName,      php_ort_tensor_getName_arginfo)
    PHP_ABSTRACT_ME(ONNX_Tensor, getType,      php_ort_tensor_getType_arginfo)
    PHP_ABSTRACT_ME(ONNX_Tensor, getTypeName,  php_ort_tensor_getTypeName_arginfo)
    PHP_ABSTRACT_ME(ONNX_Tensor, getShape,     php_ort_tensor_getShape_arginfo)
    PHP_ABSTRACT_ME(ONNX_Tensor, getSlice,     php_ort_tensor_getSlice_arginfo)
    PHP_ABSTRACT_ME(ONNX_Tensor, getData,      php_ort_tensor_getData_arginfo)

    PHP_ABSTRACT_ME(ONNX_Tensor, transpose,    php_ort_tensor_transpose_arginfo)
    PHP_FE_END
};

zend_function_entry php_ort_tensor_persistent_methods[] = {
    PHP_ME(ONNX_Tensor_Persistent, __construct,
        php_ort_tensor_persistent_construct_arginfo, ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, isPersistent, php_ort_tensor_isPersistent_arginfo, ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, getName,      php_ort_tensor_getName_arginfo,      ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, getType,      php_ort_tensor_getType_arginfo,      ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, getTypeName,  php_ort_tensor_getTypeName_arginfo,  ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, getShape,     php_ort_tensor_getShape_arginfo,     ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, getSlice,     php_ort_tensor_getSlice_arginfo,     ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, getData,      php_ort_tensor_getData_arginfo,      ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, transpose,    php_ort_tensor_transpose_arginfo,    ZEND_ACC_PUBLIC)

    PHP_ME(ONNX_Tensor, from,         php_ort_tensor_from_arginfo,         ZEND_ACC_PUBLIC|ZEND_ACC_STATIC)
    PHP_FE_END
};

zend_function_entry php_ort_tensor_transient_methods[] = {
    PHP_ME(ONNX_Tensor_Transient, __construct,
        php_ort_tensor_transient_construct_arginfo, ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, isPersistent, php_ort_tensor_isPersistent_arginfo, ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, getName,      php_ort_tensor_getName_arginfo,      ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, getType,      php_ort_tensor_getType_arginfo,      ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, getTypeName,  php_ort_tensor_getTypeName_arginfo,  ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, getShape,     php_ort_tensor_getShape_arginfo,     ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, getSlice,     php_ort_tensor_getSlice_arginfo,     ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, getData,      php_ort_tensor_getData_arginfo,      ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Tensor, transpose,    php_ort_tensor_transpose_arginfo,    ZEND_ACC_PUBLIC)

    PHP_ME(ONNX_Tensor, from,         php_ort_tensor_from_arginfo,         ZEND_ACC_PUBLIC|ZEND_ACC_STATIC)
    PHP_FE_END
};

zend_object* php_ort_tensor_create(zend_class_entry *type) {
    php_ort_tensor_t *ort = ecalloc(1,
        sizeof(php_ort_tensor_t) + zend_object_properties_size(type));

    zend_object_std_init(&ort->std, type);

    ort->std.handlers = &php_ort_tensor_handlers;
    ort->object       = NULL;

    return &ort->std;
}

static HashTable* php_ort_tensor_debug(zend_object *zo, int *temp) {
    php_ort_tensor_t *ort = php_ort_tensor_fetch(zo);
    HashTable *debug;

    ALLOC_HASHTABLE(debug);
    zend_hash_init(debug, 3, NULL, ZVAL_PTR_DTOR, 0);

    if (!ort->object) {
        goto __php_ort_tensor_debug_return;
    }

    zval persistent;

    ZVAL_BOOL(&persistent,
        ort->object->owner == PHP_ORT_OWN_HEAP);
    zend_hash_str_add(debug, 
        "persistent", sizeof("persistent")-1, 
        &persistent);

    zval type;
    
    ZVAL_LONG(&type, ort->object->type);
    zend_hash_add(debug,
        ZSTR_KNOWN(ZEND_STR_TYPE), &type);

    if (ort->object->name) {
        zval name;

        ZVAL_STR_COPY(&name, ort->object->name);
        zend_hash_add(debug,
            ZSTR_KNOWN(ZEND_STR_NAME), &name);
    }

    zval shape;

    array_init(&shape);
    for (int64_t dimension = 0; dimension < ort->object->dimensions; dimension++) {
        add_next_index_long(
            &shape, ort->object->shape[dimension]);
    }
    zend_hash_str_add(debug,
        "shape", sizeof("shape")-1, &shape);

__php_ort_tensor_debug_return:
    *temp = 1;

    return debug;
}

void php_ort_tensor_destroy(zend_object *o) {
    php_ort_tensor_t *ort =
        php_ort_tensor_fetch(o);

    ort_tensor_release(ort->object);

    zend_object_std_dtor(o);
}

PHP_MINIT_FUNCTION(ORT_TENSOR)
{
    zend_class_entry ce;

#ifdef ZTS
    php_ort_tensor_mutex = tsrm_mutex_alloc();
#endif

    zend_hash_init(&php_ort_tensors, 16, NULL, php_ort_tensor_del, 1);

    // Setup shared handlers for all tensor types
    memcpy(&php_ort_tensor_handlers,
        zend_get_std_object_handlers(), sizeof(zend_object_handlers));

    php_ort_tensor_handlers.offset = XtOffsetOf(php_ort_tensor_t, std);
    php_ort_tensor_handlers.get_debug_info = php_ort_tensor_debug;
    php_ort_tensor_handlers.free_obj = php_ort_tensor_destroy;
    php_ort_tensor_handlers.clone_obj = NULL;

    // Register the interface
    INIT_NS_CLASS_ENTRY(ce, "ONNX", "Tensor", php_ort_tensor_interface_methods);
    php_ort_tensor_interface_ce = zend_register_internal_interface(&ce);

    // Register persistent tensor class
    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Tensor", "Persistent", php_ort_tensor_persistent_methods);
    php_ort_tensor_persistent_ce = zend_register_internal_class(&ce);
    php_ort_tensor_persistent_ce->create_object = php_ort_tensor_create;
    zend_class_implements(php_ort_tensor_persistent_ce, 1, php_ort_tensor_interface_ce);

    // Register transient tensor class
    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Tensor", "Transient", php_ort_tensor_transient_methods);
    php_ort_tensor_transient_ce = zend_register_internal_class(&ce);
    php_ort_tensor_transient_ce->create_object = php_ort_tensor_create;
    zend_class_implements(php_ort_tensor_transient_ce, 1, php_ort_tensor_interface_ce);

#ifdef ZEND_ACC_NOT_SERIALIZABLE
    php_ort_tensor_persistent_ce->ce_flags |= ZEND_ACC_NOT_SERIALIZABLE;
    php_ort_tensor_transient_ce->ce_flags |= ZEND_ACC_NOT_SERIALIZABLE;
#else
    php_ort_tensor_persistent_ce->serialize = zend_class_serialize_deny;
    php_ort_tensor_persistent_ce->unserialize = zend_class_unserialize_deny;
    php_ort_tensor_transient_ce->serialize = zend_class_serialize_deny;
    php_ort_tensor_transient_ce->unserialize = zend_class_unserialize_deny;
#endif

    // Register constants on the interface
    zend_declare_class_constant_long(
        php_ort_tensor_interface_ce,
        "UNDEFINED", sizeof("UNDEFINED")-1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

    zend_declare_class_constant_long(
        php_ort_tensor_interface_ce,
        "FLOAT", sizeof("FLOAT")-1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    zend_declare_class_constant_long(
        php_ort_tensor_interface_ce,
        "DOUBLE", sizeof("DOUBLE")-1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);

    zend_declare_class_constant_long(
        php_ort_tensor_interface_ce,
        "UINT8", sizeof("UINT8")-1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    zend_declare_class_constant_long(
        php_ort_tensor_interface_ce,
        "INT8", sizeof("INT8")-1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);

    zend_declare_class_constant_long(
        php_ort_tensor_interface_ce,
        "UINT16", sizeof("UINT16")-1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);

    zend_declare_class_constant_long(
        php_ort_tensor_interface_ce,
        "INT16", sizeof("INT16")-1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);

    zend_declare_class_constant_long(
        php_ort_tensor_interface_ce,
        "UINT32", sizeof("UINT32")-1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32);

    zend_declare_class_constant_long(
        php_ort_tensor_interface_ce,
        "INT32", sizeof("INT32")-1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

    zend_declare_class_constant_long(
        php_ort_tensor_interface_ce,
        "INT64", sizeof("INT64")-1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

    zend_declare_class_constant_long(
        php_ort_tensor_interface_ce,
        "BOOL", sizeof("BOOL")-1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_TENSOR)
{
    zend_hash_destroy(&php_ort_tensors);

#ifdef ZTS
    tsrm_mutex_free(php_ort_tensor_mutex);
#endif

    return SUCCESS;
}

PHP_RINIT_FUNCTION(ORT_TENSOR)
{
    return SUCCESS;
}

PHP_RSHUTDOWN_FUNCTION(ORT_TENSOR)
{
    return SUCCESS;
}
