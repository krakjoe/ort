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

zend_class_entry* php_ort_generator_interface_ce;

void php_ort_generator_invoke(
    zval* generator,
    ONNXTensorElementDataType type,
    void* target) {
    zval rv;

    zend_fcall_info       fci = empty_fcall_info;
    zend_fcall_info_cache fcc = empty_fcall_info_cache;

    zend_fcall_info_init(
        generator, 0, &fci, &fcc, NULL, NULL);

    ZVAL_UNDEF(&rv);

    fci.retval = &rv;

    zend_call_function(&fci, &fcc);
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
    php_ort_generator_interface_ce =
        zend_register_internal_interface(&ce);

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_GENERATORS)
{
    return SUCCESS;
}