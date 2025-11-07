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

#include <php.h>

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#ifdef _WIN32
# define ORT_TLS __declspec(thread)
#else
# define ORT_TLS __thread
#endif

#include "maths/float16.h"

#ifdef HAVE_ONNXRUNTIME
# include <onnxruntime_c_api.h>
#else
typedef enum {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 1,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
} ONNXTensorElementDataType;
#endif

typedef enum {
    PHP_ORT_OWN_ZEND = 0,
    PHP_ORT_OWN_HEAP = 1
} php_ort_owner_t;

static zend_always_inline const char* php_ort_type_name(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "FLOAT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32: return "FLOAT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64: return "FLOAT64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return "INT8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return "INT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return "INT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return "INT64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return "UINT8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return "UINT16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return "UINT32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return "BOOL";
        default: return "UNKNOWN";
    }
}

static zend_always_inline size_t php_ort_type_sizeof(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return sizeof(float16);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32: return sizeof(float32);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64: return sizeof(float64);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return sizeof(int8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return sizeof(int16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return sizeof(int32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return sizeof(int64_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return sizeof(uint8_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return sizeof(uint16_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return sizeof(uint32_t);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return sizeof(uint8_t);
        default: return 0;
    }
}

static zend_always_inline uint32_t php_ort_atomic_addref(uint32_t *refcount) {
#ifdef _WIN32
    return InterlockedAdd(refcount, 1);
#elif defined(HAVE_BUILTIN_ATOMIC_CPP11)
    return __atomic_add_fetch(refcount, 1, __ATOMIC_SEQ_CST);
#else
    return __sync_add_and_fetch(refcount, 1);
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

static zend_always_inline zend_string* php_ort_string_copy(zend_string *source) {
    zend_string *dest = zend_string_alloc(ZSTR_LEN(source), 1);

    memcpy(ZSTR_VAL(dest),
           ZSTR_VAL(source),
           ZSTR_LEN(source));

    ZSTR_VAL(dest)[ZSTR_LEN(dest)] = 0;

    ZSTR_LEN(dest) = ZSTR_LEN(source);
    ZSTR_H(dest)   = ZSTR_H(source);

    GC_TYPE_INFO(dest) =
        IS_STRING |
        ((IS_STR_INTERNED | IS_STR_PERMANENT) << GC_FLAGS_SHIFT);

    return dest;
}

#ifdef HAVE_ONNXRUNTIME
extern const OrtApi* api;
#endif

PHP_MINIT_FUNCTION(ORT_CORE);
PHP_MSHUTDOWN_FUNCTION(ORT_CORE);

PHP_RINIT_FUNCTION(ORT_CORE);
PHP_RSHUTDOWN_FUNCTION(ORT_CORE);

#endif
