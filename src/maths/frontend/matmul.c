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

 #include <math.h>

 #include "status.h"

#include "maths/codegen.h"
#include "maths/core.h"

ort_math_result_t* ort_math_result_matmul(ort_tensor_t* matrix_a, ort_tensor_t* matrix_b) {
    if (!ort_math_validate_input(matrix_a, "matmul") || !ort_math_validate_input(matrix_b, "matmul")) {
        return NULL;
    }

    // Both tensors must have at least 2 dimensions (last two are the matrix dimensions)
    if (matrix_a->dimensions < 2 || matrix_b->dimensions < 2) {
        php_ort_status_throw(php_ort_status_math_error_ce, 
            "matmul: tensors must have at least 2 dimensions");
        return NULL;
    }
    
    // Get matrix dimensions (last two dimensions)
    size_t a_rows = matrix_a->shape[matrix_a->dimensions - 2];
    size_t a_cols = matrix_a->shape[matrix_a->dimensions - 1];
    size_t b_rows = matrix_b->shape[matrix_b->dimensions - 2];
    size_t b_cols = matrix_b->shape[matrix_b->dimensions - 1];
    
    // Check matrix compatibility: A[..., i, k] x B[..., k, j] -> C[..., i, j]
    if (a_cols != b_rows) {
        php_ort_status_throw(php_ort_status_math_error_ce, 
            "matmul: incompatible matrix dimensions (%zu, %zu) x (%zu, %zu)", 
            a_rows, a_cols, b_rows, b_cols);
        return NULL;
    }

    // Check batch dimensions compatibility for N-D tensors
    size_t batch_dims = matrix_a->dimensions - 2;
    if (matrix_b->dimensions - 2 != batch_dims) {
        php_ort_status_throw(php_ort_status_math_error_ce, 
            "matmul: batch dimensions must match (%zu vs %zu)", 
            batch_dims, matrix_b->dimensions - 2);
        return NULL;
    }
    
    // For N-D tensors, check that all batch dimensions are compatible
    size_t batch_size = 1;
    for (size_t i = 0; i < batch_dims; i++) {
        if (matrix_a->shape[i] != matrix_b->shape[i]) {
            php_ort_status_throw(php_ort_status_math_error_ce, 
                "matmul: batch dimension %zu mismatch (%zd vs %zd)", 
                i, matrix_a->shape[i], matrix_b->shape[i]);
            return NULL;
        }
        batch_size *= matrix_a->shape[i];
    }
    
    // Create result shape: same batch dimensions + (a_rows, b_cols)
    int64_t* result_shape = ecalloc(matrix_a->dimensions, sizeof(int64_t));
    for (size_t i = 0; i < batch_dims; i++) {
        result_shape[i] = matrix_a->shape[i];
    }
    result_shape[matrix_a->dimensions - 2] = a_rows;
    result_shape[matrix_a->dimensions - 1] = b_cols;
    
    // Allocate result tensor
    ort_tensor_t* result = ort_tensor_create_result(result_shape, matrix_a->dimensions, matrix_a->type, "matmul_result");
    efree(result_shape);
    if (!result) {
        return NULL;
    }
    
    // Perform batched matrix multiplication
    float* data_a = (float*)matrix_a->data;
    float* data_b = (float*)matrix_b->data;
    float* data_result = (float*)result->data;
    
    size_t matrix_size_a = a_rows * a_cols;
    size_t matrix_size_b = b_rows * b_cols;
    size_t matrix_size_result = a_rows * b_cols;
    
    // Process each batch
    for (size_t batch = 0; batch < batch_size; batch++) {
        float* batch_a = data_a + batch * matrix_size_a;
        float* batch_b = data_b + batch * matrix_size_b;
        float* batch_result = data_result + batch * matrix_size_result;
        
        // Standard matrix multiplication for this batch
        for (size_t i = 0; i < a_rows; i++) {
            for (size_t j = 0; j < b_cols; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < a_cols; k++) {
                    float a_val = batch_a[i * a_cols + k];
                    float b_val = batch_b[k * b_cols + j];
                    sum += a_val * b_val;
                }
                batch_result[i * b_cols + j] = sum;
            }
        }
    }
    
    return ort_math_result_create(result);
}