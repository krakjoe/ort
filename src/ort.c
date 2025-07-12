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
#include "status.h"
#include "tensor.h"
#include "maths.h"
#include "model.h"
#include "iterator.h"
#include "options.h"
#include "runtime.h"
#include "alloc.h"

const OrtApi* api;

static void __ort_alloc_default_startup(ort_alloc_t* allocator) {
    /* nothing to do */
}

static void* __ort_alloc_default_alloc(size_t size, size_t count, size_t alignment) {
    assert(
        (alignment & (alignment - 1)) == 0 && /* power of 2 */
        alignment >= sizeof(void*));          /* minimum alignment */

    void *raw = pemalloc(
        (size * count) +
        alignment - 1 + sizeof(void*), 1);

    uintptr_t address =
        (uintptr_t)raw + sizeof(void*);

    uintptr_t aligned =
        (address + (alignment - 1)) &
            ~(uintptr_t)(alignment - 1);

    void **start = (void**)aligned;

    start[-1] = raw;

    return (void*)start;
}

static void __ort_alloc_default_free(void* ptr) {
    if (ptr) {
        void *raw =
            ((void**)ptr)[-1];
        pefree(raw, 1);
    }
}

static void __ort_alloc_default_shutdown(ort_alloc_t* allocator) {
    /* nothing to do */
}

ort_alloc_t php_ort_allocator = {
    .alloc    = __ort_alloc_default_alloc,
    .free     = __ort_alloc_default_free,

    .startup  = __ort_alloc_default_startup,
    .shutdown = __ort_alloc_default_shutdown,

    .alignment = sizeof(void*) * 2,
};

PHP_MINIT_FUNCTION(ORT_CORE)
{
    ort_alloc_startup(
        &php_ort_allocator);

    api = OrtGetApiBase()
            ->GetApi(ORT_API_VERSION);;

    PHP_MINIT(ORT_ENV)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_STATUS)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_TENSOR)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_MATH)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_MODEL)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_ITERATOR)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_OPTIONS)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_RUNTIME)(INIT_FUNC_ARGS_PASSTHRU);

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_CORE)
{
    PHP_MSHUTDOWN(ORT_RUNTIME)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_OPTIONS)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_ITERATOR)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_MODEL)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_MATH)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_TENSOR)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_STATUS)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_ENV)(INIT_FUNC_ARGS_PASSTHRU);

    ort_alloc_shutdown();

    return SUCCESS;
}

PHP_RINIT_FUNCTION(ORT_CORE)
{
    PHP_RINIT(ORT_TENSOR)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_RINIT(ORT_MODEL)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_RINIT(ORT_RUNTIME)(INIT_FUNC_ARGS_PASSTHRU);

    PHP_RINIT(ORT_MATH)(INIT_FUNC_ARGS_PASSTHRU);

    return SUCCESS;
}

PHP_RSHUTDOWN_FUNCTION(ORT_CORE)
{
    PHP_RSHUTDOWN(ORT_MATH)(INIT_FUNC_ARGS_PASSTHRU);

    PHP_RSHUTDOWN(ORT_RUNTIME)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_RSHUTDOWN(ORT_MODEL)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_RSHUTDOWN(ORT_TENSOR)(INIT_FUNC_ARGS_PASSTHRU);

    return SUCCESS;
}
