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
#ifndef HAVE_ORT_ALLOC
#define HAVE_ORT_ALLOC

#include <php.h>

typedef struct _ort_alloc_t ort_alloc_t;

typedef void (*ort_alloc_startup_func_t)(ort_alloc_t* allocator);
typedef void (*ort_alloc_shutdown_func_t)(ort_alloc_t* allocator);

typedef void* (*ort_alloc_func_t)(size_t size, size_t count, size_t alignment);
typedef void* (*ort_memcpy_func_t)(void *dest, const void *src, size_t n);
typedef void (*ort_free_func_t)(void *ptr);

struct _ort_alloc_t {
    ort_alloc_func_t alloc;
    ort_memcpy_func_t memcpy;
    ort_free_func_t free;

    ort_alloc_startup_func_t startup;
    ort_alloc_shutdown_func_t shutdown;

    size_t alignment;
};

void* ort_alloc(size_t size, size_t count);
void* ort_memcpy(void *dest, const void *src, size_t n);
void ort_free(void* ptr);

void ort_alloc_align(size_t alignment);
ort_memcpy_func_t
    ort_alloc_memcpy(
        ort_memcpy_func_t memcpy);
size_t ort_alloc_alignment(void);

zend_result ZEND_MODULE_STARTUP_N(ORT_ALLOC)(
    INIT_FUNC_ARGS, ort_alloc_t* allocator);
zend_result ZEND_MODULE_SHUTDOWN_N(ORT_ALLOC)(SHUTDOWN_FUNC_ARGS);
#endif