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

/* {{{ 
    TODO(krakjoe)
        We can provide some default generators for random numbers and so on.
        Not high priority.
}}} */

#include "generators.h"

zend_class_entry* php_ort_generator_ce;
static zend_object_handlers php_ort_generator_handlers;

typedef struct _php_ort_generator_t {
    zend_fcall_info       fci;
    zend_fcall_info_cache fcc;
    zend_object           std;
} php_ort_generator_t;

static zend_always_inline php_ort_generator_t*
    php_ort_generator_fetch(zend_object* obj) {
    return (php_ort_generator_t*)
        ((char*) obj - XtOffsetOf(php_ort_generator_t, std));
}

static zend_object* php_ort_generator_create_object(
    zend_class_entry* ce) {
    php_ort_generator_t* generator = zend_object_alloc(
        sizeof(php_ort_generator_t), ce);

    zend_object_std_init(&generator->std, ce);
    object_properties_init(&generator->std, ce);

    generator->std.handlers =
        &php_ort_generator_handlers;

    memset(&generator->fci, 0, sizeof(generator->fci));
    memset(&generator->fcc, 0, sizeof(generator->fcc));

    generator->fci.size = sizeof(generator->fci);
    generator->fci.object = &generator->std;
    generator->fcc.object = &generator->std;
    generator->fcc.function_handler =
        zend_hash_find_ptr(
            &ce->function_table,
            ZSTR_KNOWN(ZEND_STR_MAGIC_INVOKE));
    generator->fcc.calling_scope = ce;
    generator->fcc.called_scope = ce;

    return &generator->std;
}

void php_ort_generator_invoke(
    zval* zv,
    ONNXTensorElementDataType type,
    void* target) {
    zval rv;
    php_ort_generator_t* generator =
        php_ort_generator_fetch(Z_OBJ_P(zv));

    ZVAL_UNDEF(&rv);
    ZVAL_COPY_VALUE(
        &generator->fci.function_name, zv);
    generator->fci.retval = &rv;

    zend_call_function(
        &generator->fci, &generator->fcc);
    php_ort_tensor_store(
        type, target, &rv);
    zval_ptr_dtor(&rv);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    php_ort_generator_invoke_arginfo, 0, 0, IS_MIXED, 0)
ZEND_END_ARG_INFO()

zend_function_entry php_ort_generator_methods[] = {
    PHP_ABSTRACT_ME(Generator, __invoke, php_ort_generator_invoke_arginfo)
    PHP_FE_END
};

PHP_MINIT_FUNCTION(ORT_GENERATORS) 
{
    zend_class_entry ce;

    INIT_NS_CLASS_ENTRY(ce, "ORT\\Tensor",
        "Generator", php_ort_generator_methods);
    php_ort_generator_ce =
        zend_register_internal_class(&ce);
    php_ort_generator_ce->create_object =
        php_ort_generator_create_object;
    
    memcpy(
        &php_ort_generator_handlers,
        zend_get_std_object_handlers(),
        sizeof(zend_object_handlers));

    php_ort_generator_handlers.offset =
        XtOffsetOf(php_ort_generator_t, std);
    php_ort_generator_handlers.clone_obj = NULL;

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_GENERATORS)
{
    return SUCCESS;
}