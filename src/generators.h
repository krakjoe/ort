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
    @file generators.h
    @brief This API provides a way to populate a tensor dynamically

    Passing a Generator into the Tensor constructor will result in the
    Generator being __invoke'd for every cell that needs a value while
    during construction.

    Note: Because the Generator is executed *during* construction, there
    is no Tensor object to pass as an argument, thus:

    interface Tensor\Generator {
        public function __invoke() : mixed;
    }

    Is the contract. __invoke may return another Generator.
    The returned Generator will be invoked until a scalar is returned.

    The resolved return value will be casted to
        the declared (at the Tensor::__construct callsite) type.
}}}*/

#ifndef HAVE_ORT_GENERATORS
#define HAVE_ORT_GENERATORS

#include "ort.h"
#include "tensor.h"

extern zend_class_entry* php_ort_generator_ce;

extern void php_ort_generator_invoke(
    zval* generator,
    ONNXTensorElementDataType type,
    void* target);

PHP_MINIT_FUNCTION(ORT_GENERATORS);
PHP_MSHUTDOWN_FUNCTION(ORT_GENERATORS);
#endif