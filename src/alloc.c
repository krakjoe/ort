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

#include "alloc.h"

static ort_alloc_t __ort_allocator = {
    .alloc    = NULL,
    .memcpy   = NULL,
    .free     = NULL,
    
    .startup  = NULL,
    .shutdown = NULL,

    .alignment = 1,
};

void ort_alloc_align(size_t alignment) {
    assert(
        (alignment & (alignment - 1)) == 0 && /* power of 2 */
        alignment >= sizeof(void*));          /* minimum alignment */

    __ort_allocator.alignment = alignment;
}

ort_memcpy_func_t ort_alloc_memcpy(ort_memcpy_func_t memcpy) {
    ort_memcpy_func_t fallback = __ort_allocator.memcpy;

    if (memcpy) {
        __ort_allocator.memcpy = memcpy;
    }

    return fallback;
}

void* ort_alloc(size_t size, size_t count) {
    return __ort_allocator.alloc(size, count,
        __ort_allocator.alignment);
}

void* ort_memcpy(void *dest, const void *src, size_t n) {
    return __ort_allocator.memcpy(dest, src, n);
}

void ort_free(void* ptr) {
    __ort_allocator.free(ptr);
}

void ort_alloc_startup(ort_alloc_t* allocator) {
    __ort_allocator = *allocator;

    if (__ort_allocator.startup) {
        __ort_allocator.startup(&__ort_allocator);
    }
}

void ort_alloc_shutdown(void)
{
    if (__ort_allocator.shutdown) {
        __ort_allocator.shutdown(&__ort_allocator);
    }
}