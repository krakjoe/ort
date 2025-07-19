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

#include "iterator.h"

#include <Zend/zend_interfaces.h>

zend_class_entry* php_ort_model_iterator_ce;
zend_object_handlers php_ort_model_iterator_handlers;

zend_object* php_ort_iterator_model(ort_model_t *model, const char* const* strings, size_t length, zval* return_value)
{
    php_ort_model_iterator_t* iterator;

    object_init_ex(return_value, php_ort_model_iterator_ce);

    iterator =
        php_ort_model_iterator_fetch(Z_OBJ_P(return_value));
    iterator->model = model;

    php_ort_atomic_addref(&model->refcount);

    iterator->data.position = 0;
    iterator->data.length   = length;
    iterator->data.strings  = strings;

    return &iterator->std;
}

static zend_object* php_ort_model_iterator_create(zend_class_entry *ce) 
{
    php_ort_model_iterator_t *iterator = ecalloc(
        1, sizeof(php_ort_model_iterator_t) + zend_object_properties_size(ce));

    zend_object_std_init(&iterator->std, ce);

    iterator->std.handlers = &php_ort_model_iterator_handlers;

    return &iterator->std;
}

void php_ort_model_iterator_destroy(zend_object *zo)
{
    php_ort_model_iterator_t* iterator =
        php_ort_model_iterator_fetch(zo);

    php_ort_model_release(iterator->model);

    zend_object_std_dtor(&iterator->std);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_iterator_rewind_arginfo, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model_Iterator, rewind)
{
    php_ort_model_iterator_t* iterator = php_ort_model_iterator_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    iterator->data.position = 0;
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_iterator_current_arginfo, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model_Iterator, current)
{
    php_ort_model_iterator_t* iterator =
        php_ort_model_iterator_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    RETURN_STRING(iterator->data.strings[iterator->data.position]);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_iterator_next_arginfo, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model_Iterator, next)
{
    php_ort_model_iterator_t* iterator =
        php_ort_model_iterator_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    iterator->data.position++;
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_iterator_key_arginfo, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model_Iterator, key)
{
    php_ort_model_iterator_t* iterator =
        php_ort_model_iterator_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    RETURN_LONG(iterator->data.position);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_iterator_valid_arginfo, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model_Iterator, valid)
{
    php_ort_model_iterator_t* iterator =
        php_ort_model_iterator_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    RETURN_BOOL(iterator->data.position < iterator->data.length);
}

zend_function_entry php_ort_model_iterator_methods[] = {
    PHP_ME(ONNX_Model_Iterator, rewind, php_ort_model_iterator_rewind_arginfo, ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model_Iterator, current, php_ort_model_iterator_current_arginfo, ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model_Iterator, next, php_ort_model_iterator_next_arginfo, ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model_Iterator, key, php_ort_model_iterator_key_arginfo, ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model_Iterator, valid, php_ort_model_iterator_valid_arginfo, ZEND_ACC_PUBLIC)

    PHP_FE_END
};

PHP_MINIT_FUNCTION(ORT_ITERATOR) 
{
    zend_class_entry ce;

    INIT_NS_CLASS_ENTRY(ce, "ORT\\Model", "Iterator", php_ort_model_iterator_methods);

    php_ort_model_iterator_ce = zend_register_internal_class(&ce);
    php_ort_model_iterator_ce->create_object = php_ort_model_iterator_create;

    zend_class_implements(
        php_ort_model_iterator_ce, 1, zend_ce_iterator);

    memcpy(&php_ort_model_iterator_handlers,
        zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    php_ort_model_iterator_handlers.offset = XtOffsetOf(php_ort_model_iterator_t, std);
    php_ort_model_iterator_handlers.free_obj = php_ort_model_iterator_destroy;
    php_ort_model_iterator_handlers.clone_obj = NULL;

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_ITERATOR) 
{
    return SUCCESS;
}