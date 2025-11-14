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

#ifndef HAVE_ORT_RUNTIME
#define HAVE_ORT_RUNTIME

#include <php.h>

#include "ort.h"
#include "model.h"
#include "options.h"

typedef struct _php_ort_runtime_t {
    ort_model_t*       model;
    zend_object        std;
} php_ort_runtime_t;

static zend_always_inline php_ort_runtime_t* php_ort_runtime_fetch(zend_object *o) {
    return (php_ort_runtime_t*) (((char*) o) - XtOffsetOf(php_ort_runtime_t, std));
}

static zend_always_inline php_ort_runtime_t* php_ort_runtime_from(zval *z) {
    return php_ort_runtime_fetch(Z_OBJ_P(z));
}

PHP_MINIT_FUNCTION(ORT_RUNTIME);
PHP_MSHUTDOWN_FUNCTION(ORT_RUNTIME);

PHP_RINIT_FUNCTION(ORT_RUNTIME);
PHP_RSHUTDOWN_FUNCTION(ORT_RUNTIME);
#endif