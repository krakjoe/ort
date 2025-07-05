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

#ifndef HAVE_ORT_ITERATOR
#define HAVE_ORT_ITERATOR

#include <php.h>

#include "model.h"

typedef struct _php_ort_model_iterator_t {
    ort_model_t *model;
    struct {
        size_t position;
        size_t length;
        const char* const* strings;
    } data;
    zend_object std;
} php_ort_model_iterator_t;

static zend_always_inline php_ort_model_iterator_t* php_ort_model_iterator_fetch(zend_object *o) {
    return (php_ort_model_iterator_t*) (((char*) o) - XtOffsetOf(php_ort_model_iterator_t, std));
}

static zend_always_inline php_ort_model_iterator_t* php_ort_model_iterator_from(zval *z) {
    return php_ort_model_iterator_fetch(Z_OBJ_P(z));
}

zend_object* php_ort_iterator_model(ort_model_t *model, const char* const* strings, size_t length, zval* return_value);

PHP_MINIT_FUNCTION(ORT_ITERATOR);
PHP_MSHUTDOWN_FUNCTION(ORT_ITERATOR);
#endif