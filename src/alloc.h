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

/*
 @brief Implements memory allocation functions for the ORT extension.
 
 There are some complicated requirements for memory allocation in this project,
 some forced upon us by onnx, some by the PHP runtime, and some by the requirements
 of the backend(s).

 Where the PHP engine expects memory to be allocated by Zend MM, we have no choice
 but to use the Zend MM allocator (so you will see emalloc/efree everywhere).

 The interactions are complex, but the contract this implementation fulfils is simple:
    Tensor memory must be allocated in a way that is compatible with the current backend.

 In all cases, including where the ORT runtime returns a tensor from inference, we must be
 sure that the above contract is fulfilled; this means that this allocator must interact with
 the ORT runtime to install itself as the allocator for ORT.
*/
#ifndef HAVE_ORT_ALLOC
#define HAVE_ORT_ALLOC

#include <php.h>

typedef struct _ort_alloc_t ort_alloc_t;

typedef void (*ort_alloc_startup_func_t)(ort_alloc_t* allocator);
typedef void (*ort_alloc_shutdown_func_t)(ort_alloc_t* allocator);

typedef void (*ort_alloc_activate_func_t)(void);
typedef void (*ort_alloc_deactivate_func_t)(void);

typedef void* (*ort_alloc_func_t)(
    size_t size, size_t count, size_t alignment);
typedef void* (*ort_memcpy_func_t)(
    void *dest, const void *src, size_t n);
typedef void (*ort_free_func_t)(void *ptr);

struct _ort_alloc_t {
    ort_alloc_func_t alloc;
    ort_memcpy_func_t memcpy_fn; /* renamed to avoid conflict with system memcpy macro */
    ort_free_func_t free;

    ort_alloc_startup_func_t startup;
    ort_alloc_shutdown_func_t shutdown;

    ort_alloc_activate_func_t activate;
    ort_alloc_deactivate_func_t deactivate;

    size_t alignment;
};

/*
 @brief shall allocate memory using the allocator's alloc function
 @param size the number of bytes to allocate
 @param count the number of elements to allocate
*/
void* ort_alloc(size_t size, size_t count);

/*
 @brief shall perform a copy using the allocator's memcpy function
 @param dest the destination pointer
 @param src the source pointer
 @param n the number of bytes to copy
*/
void* ort_memcpy(void *dest, const void *src, size_t n);

/*
 @brief shall free memory using the allocator's free function
 @param ptr the pointer to free
*/
void ort_free(void* ptr);

/*
 @brief shall set the alignment for all allocations
 @param alignment the alignment to use for all allocations
*/
void ort_alloc_align(size_t alignment);

/*
 @brief shall install the given memcpy routine for the allocator
 @param memcpy the memcpy function to be installed
 @returns the previous memcpy function
 @note must be called before any allocations are made
*/
ort_memcpy_func_t
    ort_alloc_memcpy(
        ort_memcpy_func_t memcpy);

/*
 @brief shall return the current alignment for all allocations
*/
size_t ort_alloc_alignment(void);

/*
 @brief shall return if the pointer is aligned for the current alignment
 @param ptr the pointer to check for alignment
*/
zend_bool ort_alloc_aligned(void* ptr);

/*
 @brief shall setup the allocator with the given functions
 @param activate
 @param _alloc
 @param _memcpy
 @param _free
 @param deactivate
*/
void ort_alloc_setup(
    ort_alloc_t                *backup,
    ort_alloc_activate_func_t   activate,
    ort_alloc_func_t            _alloc,
    ort_memcpy_func_t           _memcpy,
    ort_free_func_t             _free,
    ort_alloc_deactivate_func_t deactivate);

/* {{{ */
void ort_alloc_startup(void);
void ort_alloc_activate(void);
void ort_alloc_deactivate(void);
void ort_alloc_shutdown(void); /* }}} */
#endif