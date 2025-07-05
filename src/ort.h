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

#ifndef HAVE_ORT_CORE
#define HAVE_ORT_CORE

#include <onnxruntime_c_api.h>

#include <php.h>

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

typedef enum {
    PHP_ORT_OWN_ZEND = 0,
    PHP_ORT_OWN_HEAP = 1
} php_ort_owner_t;

static zend_always_inline void php_ort_atomic_addref(uint32_t *refcount) {
#ifdef _WIN32
    InterlockedAdd(refcount, 1);
#elif defined(HAVE_BUILTIN_ATOMIC_CPP11)
    __atomic_add_fetch(refcount, 1, __ATOMIC_SEQ_CST);
#else
    __sync_add_and_fetch(refcount, 1);
#endif
}

static zend_always_inline uint32_t php_ort_atomic_delref(uint32_t *refcount) {
#ifdef _WIN32
    return InterlockedAdd(refcount, -1);
#elif defined(HAVE_BUILTIN_ATOMIC_CPP11)
    return __atomic_sub_fetch(refcount, 1, __ATOMIC_SEQ_CST);
#else
    return __sync_sub_and_fetch(refcount, 1);
#endif
}

#ifdef _WIN32
/* {{{ Windows-specific functions }}}*/
/* {{{
    Note: we need to convert UTF-8 strings to UTF-16 (wchar_t*) for ONNX Runtime API calls.
}}}*/
#include <windows.h>
#include <stdlib.h>

/** {{{ Temporary strings must only be returned by php_ort_string_pass,
        the caller must free them with php_ort_string_free() }}} */
typedef wchar_t* php_ort_string_temp_t;

// Converts a UTF-8 C string to a newly allocated wchar_t* (UTF-16).
static zend_always_inline php_ort_string_temp_t php_ort_string_pass(const char* utf8)
{
    if (!utf8)
        return NULL;
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);

    if (wlen == 0) {
        return NULL; // Conversion failed
    }

    wchar_t* wide = (wchar_t*)ecalloc(wlen, sizeof(wchar_t));

    if (MultiByteToWideChar(
            CP_UTF8, 0, utf8, -1, wide, wlen) == 0) {
        efree(wide);
        return NULL;
    }

    return wide;
}
// Frees a wchar_t* allocated by php_ort_string_pass
static zend_always_inline void php_ort_string_free(php_ort_string_temp_t wide) {
    if (wide) {
        efree(wide);
    }
}
#else
typedef char* php_ort_string_temp_t;

static zend_always_inline php_ort_string_temp_t php_ort_string_pass(const char* utf8) {
    return (php_ort_string_temp_t) utf8; // No conversion needed on non-Windows platforms
}
static zend_always_inline void php_ort_string_free(php_ort_string_temp_t utf8) {
    // Don't need to free because we don't allocate memory for utf8
}
#endif

extern const OrtApi* api;

PHP_MINIT_FUNCTION(ORT_CORE);
PHP_MSHUTDOWN_FUNCTION(ORT_CORE);

PHP_RINIT_FUNCTION(ORT_CORE);
PHP_RSHUTDOWN_FUNCTION(ORT_CORE);

#endif
