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

#ifndef ORT_MATHS_POOL
#define ORT_MATHS_POOL

#include "ort.h"

typedef void (*ort_task_func_t)(
    void *arg, size_t index, size_t count);

typedef struct _ort_pool_ctx_layout_t {
    size_t element;
    size_t total;
    size_t chunk;
} ort_pool_ctx_layout_t;

typedef struct _ort_pool_binary_ctx_t {
    ort_pool_ctx_layout_t layout;
    void *result;
    void *a;
    void *b;
    void (*op)(void *result, const void *a, const void *b, size_t n);
} ort_pool_binary_ctx_t;

typedef struct _ort_pool_unary_ctx_t {
    ort_pool_ctx_layout_t layout;
    void *result;
    void *a;
    void (*op)(void *result, const void *a, size_t n);
} ort_pool_unary_ctx_t;

typedef struct _ort_pool_scalar_ctx_t {
    ort_pool_ctx_layout_t layout;
    void *result;
    void *a;
    void *b;
    void (*op)(void *result, const void *a, const void *b, size_t n);
} ort_pool_scalar_ctx_t;

typedef struct _ort_pool_reduce_tensor_ctx_t {
    ort_pool_ctx_layout_t layout;
    void *result;
    const void *a;
    size_t elements;
    void (*op)(void *result, const void *a, size_t n);
} ort_pool_reduce_tensor_ctx_t;

typedef struct _ort_reduce_axis_layout_t {
    size_t element;      // size of one element in bytes
    size_t total;        // total number of output elements (outer * inner)
    size_t chunk;        // chunk size for thread pool
    size_t axis_size;    // length of reduction axis
    size_t outer;        // product of dims before axis
    size_t inner;        // product of dims after axis
} ort_reduce_axis_layout_t;

typedef struct _ort_pool_reduce_axis_ctx_t {
    ort_reduce_axis_layout_t layout; // all scheduling/precomputed info
    void *result;                    // output data pointer
    const void *a;                   // input data pointer
    void (*op)(void *result, const void *a, size_t outer, size_t axis, size_t inner); // reduction kernel
} ort_pool_reduce_axis_ctx_t;

typedef struct _ort_pool_matmul_ctx_t {
    ort_pool_ctx_layout_t layout; // chunking info: total = batch_size, chunk = batches per thread
    void *result;                 // pointer to result buffer (all batches)
    const void *a;                // pointer to input A buffer (all batches)
    const void *b;                // pointer to input B buffer (all batches)
    size_t a_rows;                // rows in A (per batch)
    size_t a_cols;                // cols in A (per batch)
    size_t b_cols;                // cols in B (per batch)
    size_t type_size;             // size of one element in bytes
    size_t matrix_size_a;         // elements in one A matrix
    size_t matrix_size_b;         // elements in one B matrix
    size_t matrix_size_result;    // elements in one result matrix
    void (*op)(void *result, const void *a, const void *b, size_t a_rows, size_t a_cols, size_t b_cols); // matmul kernel
} ort_pool_matmul_ctx_t;

int ort_pool_init(size_t size);
void ort_pool_submit(
    ort_task_func_t func, void *arg, size_t count);
void ort_pool_destroy(void);

void ort_pool_binary_worker(void *arg, size_t index, size_t count);
void ort_pool_unary_worker(void *arg, size_t index, size_t count);
void ort_pool_scalar_worker(void *arg, size_t index, size_t count);

void ort_pool_reduce_tensor_worker(void *arg, size_t index, size_t count);
void ort_pool_reduce_axis_worker(void *arg, size_t index, size_t count);

void ort_pool_matmul_worker(void *arg, size_t index, size_t count);

/**
 * Sets the scale (number of threads) to use for scheduling
 * @returns the scale (number of threads) at call time, or zero on error
 * @note the caller must restore the scale
 */
size_t ort_pool_scale(size_t cores);

/**
 * Get the scale (number of threads) to use for scheduling
 * @returns the number of threads available to the pool (currently)
 */
size_t ort_pool_cores(void);

/**
 * Get the maximum number of threads supported by the pool
 * @returns the maximum number of threads supported by the pool
 */
size_t ort_pool_max(void);
#endif
