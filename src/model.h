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

#ifndef HAVE_ORT_MODEL
#define HAVE_ORT_MODEL

#include <php.h>

#include "ort.h"

extern zend_class_entry* php_ort_model_ce;

typedef enum _ort_model_source_kind_t {
  ORT_MODEL_SOURCE_FILE,
  ORT_MODEL_SOURCE_MEMORY,
} ort_model_source_kind_t;

typedef struct _ort_model_t {
  uint32_t                  refcount;
  ort_model_source_kind_t   kind;
  union {
    zend_string*            file;
    zend_string*            memory;  
  } source;
  zend_string*              name;
  OrtModelMetadata*         metadata;       /* Raw metadata */
  OrtSession*               session;
  OrtAllocator*             allocator;
  struct {
    size_t                  input;
    size_t                  output;
  } counters;
  struct {
    char**                  input;
    char**                  output;
  } names;
} ort_model_t;

typedef struct _php_ort_model_t {
    ort_model_t              *object;
    zend_object               std;
} php_ort_model_t;

static zend_always_inline php_ort_model_t* php_ort_model_fetch(zend_object *o) {
    return (php_ort_model_t*) (((char*) o) - XtOffsetOf(php_ort_model_t, std));
}

static zend_always_inline php_ort_model_t* php_ort_model_from(zval *z) {
    return php_ort_model_fetch(Z_OBJ_P(z));
}

PHP_MINIT_FUNCTION(ORT_MODEL);
PHP_MSHUTDOWN_FUNCTION(ORT_MODEL);

PHP_RINIT_FUNCTION(ORT_MODEL);
PHP_RSHUTDOWN_FUNCTION(ORT_MODEL);

void php_ort_model_release(ort_model_t *model);

#endif
