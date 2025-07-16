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
#include "alloc.h"

struct _ort_alloc_t {
    ort_alloc_func_t alloc;
    ort_memcpy_func_t memcpy;
    ort_free_func_t free;

    ort_alloc_startup_func_t startup;
    ort_alloc_shutdown_func_t shutdown;

    size_t alignment;
};

static OrtMemoryInfo* minfo;

const OrtMemoryInfo* __ort_alloc_default_info_for_onnx(
    const OrtAllocator* allocator) {
    if (!minfo) {
        OrtStatus* status =
            api->CreateCpuMemoryInfo(
                OrtDeviceAllocator, OrtMemTypeDefault, &minfo);
        if (status) {
            fprintf(stderr,
                "ort: failed to create default memory info: %s\n",
                api->GetErrorMessage(status));

            api->ReleaseStatus(status);
            exit(EXIT_FAILURE);
        }
    }
    return minfo;
}

static void* __ort_alloc_default_malloc_for_onnx(
    OrtAllocator* allocator, size_t size) {
    return ort_alloc(size, 1);
}

static void __ort_alloc_default_free_for_onnx(
    OrtAllocator* allocator, void* ptr) {
    ort_free(ptr);
}

static void* __ort_alloc_default_reserve_for_onnx(
    OrtAllocator* allocator, size_t size) {
    return ort_alloc(size, 1);
}

static const OrtAllocator __ort_alloc_default_for_onnx = {
    .Alloc   = __ort_alloc_default_malloc_for_onnx,
    .Reserve = __ort_alloc_default_reserve_for_onnx,
    .Free    = __ort_alloc_default_free_for_onnx,
    .Info    = __ort_alloc_default_info_for_onnx,
    .version = ORT_API_VERSION,
};

static void __ort_alloc_default_startup(ort_alloc_t* allocator) {
    OrtStatus* status =
        api->RegisterAllocator(
            php_ort_environment(),
            (OrtAllocator*)
                &__ort_alloc_default_for_onnx);
    if (status) {
        fprintf(stderr,
            "ort: failed to register allocator: %s\n",
            api->GetErrorMessage(status));

        api->ReleaseStatus(status);
        exit(EXIT_FAILURE);
    }
}

static void* __ort_alloc_default_alloc(size_t size, size_t count, size_t alignment) {
    assert(
        (alignment & (alignment - 1)) == 0 && /* power of 2 */
        alignment >= sizeof(void*));          /* minimum alignment */

    void *raw = pemalloc(
        (size * count) + alignment - 1 + sizeof(void*), 1);

    uintptr_t address = (uintptr_t)raw + sizeof(void*);
    
    // Ensure we have space for the raw pointer before alignment
    uintptr_t aligned = (address + alignment - 1) & ~(uintptr_t)(alignment - 1);
    
    // If there's not enough space before aligned address, move to next boundary
    if (aligned - (uintptr_t)raw < sizeof(void*)) {
        aligned += alignment;
    }

    void **start = (void**)aligned;
    start[-1] = raw;

    return (void*)start;
}

static void* __ort_alloc_default_memcpy(void *dest, const void *src, size_t n) {
    if (n == 0) {
        return dest;
    }
    return memcpy(dest, src, n);
}

static void __ort_alloc_default_free(void* ptr) {
    if (ptr) {
        void *raw =
            ((void**)ptr)[-1];
        pefree(raw, 1);
    }
}

static void __ort_alloc_default_shutdown(ort_alloc_t* allocator) {}

ORT_TLS ort_alloc_t __ort_allocator = {
    .alloc    = __ort_alloc_default_alloc,
    .memcpy   = __ort_alloc_default_memcpy,
    .free     = __ort_alloc_default_free,

    .startup  = __ort_alloc_default_startup,
    .shutdown = __ort_alloc_default_shutdown,

    .alignment = sizeof(void*) * 2,
};

void ort_alloc_align(size_t alignment) {
    assert(
        (alignment & (alignment - 1)) == 0 && /* power of 2 */
        alignment >= sizeof(void*));          /* minimum alignment */

    __ort_allocator.alignment = alignment;
}

size_t ort_alloc_alignment(void) {
    return __ort_allocator.alignment;
}

zend_bool ort_alloc_aligned(void* ptr) {
    return (zend_bool)
        ((uintptr_t)ptr % 
            __ort_allocator.alignment) == 0;
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

PHP_MINIT_FUNCTION(ORT_ALLOC) 
{
    if (__ort_allocator.startup) {
        __ort_allocator.startup(&__ort_allocator);
    }

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_ALLOC)
{
    if (__ort_allocator.shutdown) {
        __ort_allocator.shutdown(&__ort_allocator);
    }

    return SUCCESS;
}