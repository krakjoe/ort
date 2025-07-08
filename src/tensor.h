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

#ifndef HAVE_ORT_TENSOR
#define HAVE_ORT_TENSOR

#include <php.h>

#include "ort.h"

extern zend_class_entry *php_ort_tensor_interface_ce;
extern zend_class_entry *php_ort_tensor_persistent_ce;
extern zend_class_entry *php_ort_tensor_transient_ce;

typedef struct _ort_tensor_t ort_tensor_t;

struct _ort_tensor_t {
    uint32_t                  refcount;
    php_ort_owner_t           owner;
    OrtValue*                 value;
    ort_tensor_t*             parent;
    zend_string*              name;
    ONNXTensorElementDataType type;
    int64_t*                  shape;
    size_t                    dimensions;
    size_t                    elements;
    void*                     data;
};

typedef struct _php_ort_tensor_t {
    ort_tensor_t              *object;
    zend_object               std;
} php_ort_tensor_t;

static zend_always_inline php_ort_tensor_t* php_ort_tensor_fetch(zend_object *o) {
    return (php_ort_tensor_t*) (((char*) o) - XtOffsetOf(php_ort_tensor_t, std));
}

static zend_always_inline php_ort_tensor_t* php_ort_tensor_from(zval *z) {
    return php_ort_tensor_fetch(Z_OBJ_P(z));
}

static zend_always_inline size_t php_ort_tensor_sizeof(ort_tensor_t *tensor) {
    return php_ort_type_sizeof(tensor->type);
}

void ort_tensor_release(ort_tensor_t *tensor);

OrtValue*     php_ort_tensor_value(php_ort_tensor_t* ort);
ort_tensor_t* php_ort_tensor_object(OrtValue* value);

PHP_MINIT_FUNCTION(ORT_TENSOR);
PHP_MSHUTDOWN_FUNCTION(ORT_TENSOR);

PHP_RINIT_FUNCTION(ORT_TENSOR);
PHP_RSHUTDOWN_FUNCTION(ORT_TENSOR);

#endif
