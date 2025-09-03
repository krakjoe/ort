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

#ifndef HAVE_ORT_OPTIONS
#define HAVE_ORT_OPTIONS

#include <php.h>

#include "ort.h"

extern zend_class_entry* php_ort_options_ce;

typedef struct _ort_options_t {
    uint32_t           refcount;
    php_ort_owner_t    owner;
    OrtSessionOptions* options;
} ort_options_t;

typedef struct _php_ort_options_t {
    ort_options_t*     object;
    zend_object        std;
} php_ort_options_t;

static zend_always_inline php_ort_options_t* php_ort_options_fetch(zend_object *o) {
    return (php_ort_options_t*) (((char*) o) - XtOffsetOf(php_ort_options_t, std));
}

static zend_always_inline php_ort_options_t* php_ort_options_from(zval *z) {
    return php_ort_options_fetch(Z_OBJ_P(z));
}

extern ort_options_t*
    php_ort_options_default(void);
extern void php_ort_options_release(ort_options_t *options);

PHP_MINIT_FUNCTION(ORT_OPTIONS);
PHP_MSHUTDOWN_FUNCTION(ORT_OPTIONS);
#endif
