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
#include "ort.h"

#include "env.h"

#ifdef HAVE_ONNXRUNTIME
static OrtEnv* __php_ort_environment = NULL;

OrtEnv* php_ort_environment(void) {
    return __php_ort_environment;
}
#else
void* php_ort_environment(void) {
    return NULL;
}
#endif

PHP_MINIT_FUNCTION(ORT_ENV) 
{
#ifdef HAVE_ONNXRUNTIME
    return api->CreateEnv(
        ORT_LOGGING_LEVEL_WARNING, 
        "[ORT PHP]", &__php_ort_environment) ?
        FAILURE : SUCCESS;
#endif
}

PHP_MSHUTDOWN_FUNCTION(ORT_ENV) 
{
#ifdef HAVE_ONNXRUNTIME
    if (__php_ort_environment != NULL) {
        api->ReleaseEnv(
            __php_ort_environment);
    }
#endif
    return SUCCESS;
}