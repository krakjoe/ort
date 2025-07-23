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
#include "alloc.h"
#include "maths/core.h"

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
    ort_math_kernel_binary_t op;
} ort_pool_binary_ctx_t;

typedef struct _ort_pool_unary_ctx_t {
    ort_pool_ctx_layout_t layout;
    void *result;
    void *a;
    ort_math_kernel_unary_t op;
} ort_pool_unary_ctx_t;

typedef struct _ort_pool_scalar_ctx_t {
    ort_pool_ctx_layout_t layout;
    void *result;
    void *a;
    void *b;
    ort_math_kernel_scalar_t op;
} ort_pool_scalar_ctx_t;

typedef struct _ort_pool_reduce_tensor_ctx_t {
    ort_pool_ctx_layout_t layout;
    void *result;
    const void *a;
    size_t elements;
    ort_math_kernel_reduce_tensor_t op;
} ort_pool_reduce_tensor_ctx_t;

typedef struct _ort_pool_reduce_axis_ctx_t {
    ort_pool_ctx_layout_t layout;     // chunking info: element size, total, chunk
    void *result;                     // output data pointer
    const void *a;                    // input data pointer
    const size_t *input_shape;        // input shape
    size_t input_dims;                // input dimensions
    const size_t *output_shape;       // output shape
    size_t output_dims;               // output dimensions
    size_t axis;                      // axis to reduce
    ort_math_kernel_reduce_axis_t op; // reduction kernel
} ort_pool_reduce_axis_ctx_t;

typedef struct _ort_pool_matmul_ctx_t {
    ort_pool_ctx_layout_t layout; // chunking info: total = number of output rows, chunk = rows per thread
    void *result;                 // pointer to result buffer (output matrix, row-major)
    const void *a;                // pointer to input A buffer (row-major, one row per kernel call)
    const void *b;                // pointer to input B buffer (row-major)
    size_t a_cols;                // number of columns in A (width of input row)
    size_t b_cols;                // number of columns in B (width of output row)
    size_t type_size;             // size of one element in bytes
    size_t matrix_size_a;         // number of elements in one A matrix
    size_t matrix_size_b;         // number of elements in one B matrix
    size_t matrix_size_result;    // number of elements in one result matrix 
    ort_math_kernel_matmul_t op;  // matmul kernel: computes one output row per call
} ort_pool_matmul_ctx_t;

typedef struct _ort_pool_cast_ctx_t {
    const void* src;
    void* dst;
    ONNXTensorElementDataType src_type;
    ONNXTensorElementDataType dst_type;
    size_t count;
    void (*op)(const void* src, void* dst, ONNXTensorElementDataType src_type, ONNXTensorElementDataType dst_type);
} ort_pool_cast_ctx_t;

int ort_pool_init(size_t size);
void ort_pool_submit(
    ort_task_func_t func, void *arg, size_t count);
void ort_pool_destroy(void);

void ort_pool_cast_worker(void *arg, size_t index, size_t count);

void ort_pool_binary_worker(void *arg, size_t index, size_t count);
void ort_pool_unary_worker(void *arg, size_t index, size_t count);
void ort_pool_scalar_worker(void *arg, size_t index, size_t count);

void ort_pool_reduce_tensor_worker(void *arg, size_t index, size_t count);
void ort_pool_reduce_axis_worker(void *arg, size_t index, size_t count);

void ort_pool_matmul_worker(void *arg, size_t index, size_t count);

/* {{{ Scaling stuff */
#define ORT_SCALE_THRESHOLD 300000

typedef enum {
    ORT_POOL_SCALE_CORES     = 1,
    ORT_POOL_SCALE_THRESHOLD = 2,
    ORT_POOL_SCALE_ERROR     = 16,
} ort_pool_scale_kind_t;

typedef struct _ort_pool_scale_t {
    ort_pool_scale_kind_t kind;
    zend_long             cores; // number of cores to use
    zend_long             threshold; // threshold for parallelization
} ort_pool_scale_t;

/**
 * Sets the scale to use for scheduling
 * @returns the scale at call time
 * @note the caller must restore the scale
 */
ort_pool_scale_t ort_pool_scale(ort_pool_scale_t* scale);
/* }}} */

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

/**
 * Get the current parallelization threshold
 * @returns the threshold at call time
 */
size_t ort_pool_threshold();

/*
 * Compute chunk size and number of chunks for parallel work, ensuring
 * each chunk start is aligned according to requirements of the allocator.
 *
 * @param total      Total number of elements to process
 * @param size       Size of each element in bytes
 * @param chunk      [out] Receives chunk size in elements
 * @return           Number of chunks to use
 */
static zend_always_inline size_t ort_pool_chunk(
    size_t total,
    size_t size,
    size_t *chunk
) {
    // Apply threshold
    if (total <= ort_pool_threshold()) {
        if (chunk) {
            *chunk = total; // No parallelization needed,
                            // use all elements as one chunk
        }
        return 1; // Only one chunk needed
    }

    // Elements per alignment boundary
    size_t elements = ort_alloc_alignment() / size;
    if (elements == 0)
        elements = 1;

    // Start with a chunk size that divides the work among cores
    size_t chunking =
        (total + ort_pool_cores() - 1) 
            / ort_pool_cores();

    // Round up chunk size to next multiple of elements
    chunking =
        ((chunking + elements - 1) / elements)
            * elements;

    // Don't exceed total elements
    if (chunking > total)
        chunking = total;

    if (chunk)
        *chunk = chunking;

    return (total + chunking - 1) / chunking;
}

#endif
