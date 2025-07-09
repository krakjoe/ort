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

#include "maths/cast.h"
#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/result.h"

// Matrix multiplication for a single batch (C = A x B)
#define ORT_MATH_MATMUL_IMPL_FOR_TYPE(c_type, unused)  \
    ORT_MATH_FRONTEND_MATMUL_OP_DECL(c_type) {         \
        const c_type* va = (const c_type*)a;           \
        const c_type* vb = (const c_type*)b;           \
        c_type* res = (c_type*)result;                 \
        for (size_t i = 0; i < a_rows; i++) {          \
            for (size_t j = 0; j < b_cols; j++) {      \
                int64_t sum = 0;                       \
                for (size_t k = 0; k < a_cols; k++) {  \
                    sum += va[                         \
                        i * a_cols + k                 \
                    ] * vb[k * b_cols + j];            \
                }                                      \
                /* Clamp for small integer types */    \
                if (sizeof(c_type) == 1) {                                 \
                    if ((c_type)-128 == (int8_t)-128) { /* int8_t */       \
                        if (sum > INT8_MAX) sum = INT8_MAX;                \
                        else if (sum < INT8_MIN) sum = INT8_MIN;           \
                    } else { /* uint8_t */                                 \
                        if (sum > UINT8_MAX) sum = UINT8_MAX;              \
                        else if (sum < 0) sum = 0;                         \
                    }                                                      \
                } else if (sizeof(c_type) == 2) {                          \
                    if ((c_type)-32768 == (int16_t)-32768) { /* int16_t */ \
                        if (sum > INT16_MAX) sum = INT16_MAX;              \
                        else if (sum < INT16_MIN) sum = INT16_MIN;         \
                    } else { /* uint16_t */                                \
                        if (sum > UINT16_MAX) sum = UINT16_MAX;            \
                        else if (sum < 0) sum = 0;                         \
                    }                                                      \
                }                                                          \
                res[i * b_cols + j] = (c_type)sum;                         \
            }                                                              \
        }                                                                  \
    }

ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_MATMUL_IMPL_FOR_TYPE)

static ort_math_element_op_func_t ort_math_frontend_get_matmul_func(ONNXTensorElementDataType type) {
    const ort_math_dispatch_t* dispatch =
        ort_math_dispatch_type(type);
    return (void*) dispatch->matmul_func;
}

ort_tensor_t* ort_math_result_matmul(ort_tensor_t* matrix_a, ort_tensor_t* matrix_b) {
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

    // Strict ONNX/NumPy-style type promotion for matmul (use struct for richer info)
    ort_math_type_promotion_t promotion = ort_math_type_promote_strict(matrix_a->type, matrix_b->type);
    if (!promotion.is_valid) {
        php_ort_status_throw(php_ort_status_math_error_ce,
            "matmul: unsupported type promotion (%s x %s)",
            php_ort_type_name(matrix_a->type), php_ort_type_name(matrix_b->type));
        return NULL;
    }
    ONNXTensorElementDataType promoted_type = promotion.result_type;

    // Create result shape: same batch dimensions + (a_rows, b_cols)
    int64_t* result_shape = ecalloc(matrix_a->dimensions, sizeof(int64_t));
    for (size_t i = 0; i < batch_dims; i++) {
        result_shape[i] = matrix_a->shape[i];
    }
    result_shape[matrix_a->dimensions - 2] = a_rows;
    result_shape[matrix_a->dimensions - 1] = b_cols;

    // Allocate result tensor with promoted type
    ort_tensor_t* result = ort_math_result_tensor(
        result_shape,
        matrix_a->dimensions,
        promoted_type, "matmul_result");

    efree(result_shape);

    // Get the correct matmul kernel for the promoted type
    ort_math_matmul_op_func_t operation = 
        (ort_math_matmul_op_func_t)
            ort_math_frontend_get_matmul_func(promoted_type);

    // Prepare pointers for each batch, with type casting if needed
    size_t matrix_size_a = a_rows * a_cols;
    size_t matrix_size_b = b_rows * b_cols;
    size_t matrix_size_result = a_rows * b_cols;

    void* tmp_a = NULL;
    void* tmp_b = NULL;
    int need_cast_a = (matrix_a->type != promoted_type);
    int need_cast_b = (matrix_b->type != promoted_type);
    size_t type_size = php_ort_type_sizeof(promoted_type);

    for (size_t batch = 0; batch < batch_size; batch++) {
        const void* batch_a =
            (const char*)matrix_a->data + 
                batch * matrix_size_a * php_ort_type_sizeof(matrix_a->type);
        const void* batch_b =
            (const char*)matrix_b->data +
                batch * matrix_size_b * php_ort_type_sizeof(matrix_b->type);
        void* batch_result =
            (char*)result->data +
                batch * matrix_size_result * type_size;

        // Cast A if needed
        if (need_cast_a) {
            if (!tmp_a) tmp_a = emalloc(matrix_size_a * type_size);
            ort_math_cast_buffer(
                batch_a, tmp_a, 
                matrix_a->type, promoted_type, matrix_size_a);
            batch_a = tmp_a;
        }
        // Cast B if needed
        if (need_cast_b) {
            if (!tmp_b) tmp_b = emalloc(matrix_size_b * type_size);
            ort_math_cast_buffer(
                batch_b, tmp_b,
                matrix_b->type, promoted_type, matrix_size_b);
            batch_b = tmp_b;
        }

        operation(batch_result, batch_a, batch_b, a_rows, a_cols, b_cols);
    }
    if (tmp_a) {
        efree(tmp_a);
    }
    if (tmp_b) {
        efree(tmp_b);
    }

    return result;
}