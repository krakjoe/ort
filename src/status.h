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

#ifndef HAVE_ORT_STATUS
#define HAVE_ORT_STATUS
 
#include <php.h>

#include <Zend/zend_exceptions.h>

extern zend_class_entry* php_ort_status_error_ce;

#ifdef ZTS
extern zend_class_entry* php_ort_status_safetyerror_ce;
#endif

extern zend_class_entry* php_ort_status_scaling_error_ce;

extern zend_class_entry* php_ort_status_tensor_error_ce;
extern zend_class_entry* php_ort_status_tensor_notfound_ce;
extern zend_class_entry* php_ort_status_tensor_invalidtype_ce;
extern zend_class_entry* php_ort_status_tensor_invalidshape_ce;
extern zend_class_entry* php_ort_status_tensor_invaliddata_ce;
extern zend_class_entry* php_ort_status_tensor_invalidmemory_ce;
extern zend_class_entry* php_ort_status_tensor_invalidoffset_ce;
extern zend_class_entry* php_ort_status_tensor_invalidaccess_ce;

// Mathematical operation exceptions
extern zend_class_entry* php_ort_status_math_error_ce;
extern zend_class_entry* php_ort_status_math_invalidshape_ce;
extern zend_class_entry* php_ort_status_math_invalidtype_ce;
extern zend_class_entry* php_ort_status_math_invalidbroadcast_ce;
extern zend_class_entry* php_ort_status_math_invalidoperation_ce;
extern zend_class_entry* php_ort_status_math_divisionbyzero_ce;

// Model exceptions
extern zend_class_entry* php_ort_status_model_error_ce;
extern zend_class_entry* php_ort_status_model_invalidmemory_ce;
extern zend_class_entry* php_ort_status_model_invalidoptions_ce;
extern zend_class_entry* php_ort_status_model_invalidmodel_ce;
extern zend_class_entry* php_ort_status_model_invalidinput_ce;
extern zend_class_entry* php_ort_status_model_invalidoutput_ce;
extern zend_class_entry* php_ort_status_model_invalidindex_ce;
extern zend_class_entry* php_ort_status_model_invalidfile_ce;

// Schema exceptions
extern zend_class_entry* php_ort_status_schema_error_ce;
extern zend_class_entry* php_ort_status_schema_invalidsymbol_ce;
extern zend_class_entry* php_ort_status_schema_invalidarguments_ce;

// TODO(krakjoe) this
extern zend_class_entry* php_ort_status_model_runtimeexception_ce;

static zend_object* php_ort_status_throw(zend_class_entry* ce, const char *format, ...) {
    va_list args;

    va_start(args, format);
    zend_string *formatted = 
        zend_vstrpprintf(0, format, args);
    va_end(args);

    zend_object *result =
        zend_throw_exception(
            ce, ZSTR_VAL(formatted), 0);

    zend_string_release(formatted);
    return result;
}

#define php_ort_status_flow(status, flow, ce, format, ...) do  \
{                                                              \
    if (status) {                                              \
        if (!EG(exception)) {                                  \
            php_ort_status_throw(                              \
                ce, format, ##__VA_ARGS__);                    \
        }                                                      \
        flow;                                                  \
    }                                                          \
} while(0)

PHP_MINIT_FUNCTION(ORT_STATUS);
PHP_MSHUTDOWN_FUNCTION(ORT_STATUS);
#endif