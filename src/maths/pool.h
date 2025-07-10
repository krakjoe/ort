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

typedef struct _ort_pool_slow_binary_ctx_t {
    ort_pool_ctx_layout_t layout;
    void *result;
    void *a;
    void *b;
    int64_t *result_shape;
    size_t result_dimensions;
    int64_t *a_shape;
    size_t a_dimensions;
    int64_t *b_shape;
    size_t b_dimensions;
    ONNXTensorElementDataType a_type;
    ONNXTensorElementDataType b_type;
    ONNXTensorElementDataType result_type;
    size_t element_size;
    size_t a_element_size;
    size_t b_element_size;
    size_t a_dim_offset;
    size_t b_dim_offset;
    void (*op)(void *result, const void *a, const void *b, size_t n);
} ort_pool_slow_binary_ctx_t;

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

size_t ort_pool_cores(void);

int ort_pool_init(size_t size);
void ort_pool_submit(
    ort_task_func_t func, void *arg, size_t count);
void ort_pool_destroy(void);

void ort_pool_binary_worker(void *arg, size_t index, size_t count);
void ort_pool_unary_worker(void *arg, size_t index, size_t count);
void ort_pool_scalar_worker(void *arg, size_t index, size_t count);
void ort_pool_slow_binary_worker(void *arg, size_t index, size_t count);

void ort_pool_reduce_tensor_worker(void *arg, size_t index, size_t count);
void ort_pool_reduce_axis_worker(void *arg, size_t index, size_t count);
#endif
