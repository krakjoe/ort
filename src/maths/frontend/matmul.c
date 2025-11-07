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
 @brief Implements frontend matmul operations for tensors
 @test tests/math/matmul
*/

#include <math.h>

#include "status.h"

#include "alloc.h"
#include "maths/cast.h"
#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/pool.h"
#include "maths/result.h"
#include "maths/schema/matmul.h"

// Matrix multiplication for a single batch (C = A x B)

ORT_MATH_FRONTEND_MATMUL_OP_DECL(float16) {
    const float16* va = (const float16*)a;
    const float16* vb = (const float16*)b;
    float16* res = (float16*)result;
    for (size_t j = 0; j < b_cols; j++) {
        float32 sum = 0;
        for (size_t k = 0; k < a_cols; k++) {
            sum += ort_math_float32_from_float16(va[k]) *
                   ort_math_float32_from_float16(vb[k * b_cols + j]);
        }
        res[j] = ort_math_float16_from_float32(sum);
    }
}

#define ORT_MATH_MATMUL_IMPL_FOR_REAL_TYPE(c_type, onnx_type)  \
    ORT_MATH_FRONTEND_MATMUL_OP_DECL(c_type) {            \
        const c_type* va = (const c_type*)a;              \
        const c_type* vb = (const c_type*)b;              \
        c_type* res = (c_type*)result;                    \
        for (size_t j = 0; j < b_cols; j++) {             \
            c_type sum = 0;                               \
            for (size_t k = 0; k < a_cols; k++) {         \
                sum += va[k] * vb[k * b_cols + j];        \
            }                                             \
            res[j] = (c_type) sum;                        \
        }                                                 \
    }

/* TODO(krakjoe) fix accumulator type to be consistent with backend */
#define ORT_MATH_MATMUL_IMPL_FOR_INTEGER_TYPE(c_type, onnx_type)  \
    ORT_MATH_FRONTEND_MATMUL_OP_DECL(c_type) {            \
        const c_type* va = (const c_type*)a;              \
        const c_type* vb = (const c_type*)b;              \
        c_type* res = (c_type*)result;                    \
        for (size_t j = 0; j < b_cols; j++) {             \
            int64_t sum = 0;                              \
            for (size_t k = 0; k < a_cols; k++) {         \
                sum += va[k] * vb[k * b_cols + j];        \
            }                                             \
            if ((onnx_type ==                             \
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)) {       \
                sum = ORT_MATH_CLAMP(sum, INT8, int8_t);         \
            } else if ((onnx_type ==                             \
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)) {      \
                sum = ORT_MATH_CLAMP_MAX(sum, UINT8, uint8_t);   \
            } else if ((onnx_type ==                             \
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)) {      \
                sum = ORT_MATH_CLAMP(sum, INT16, int16_t);       \
            } else if ((onnx_type ==                             \
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {     \
                sum = ORT_MATH_CLAMP_MAX(sum, UINT16, uint16_t); \
            }  else if ((onnx_type ==                            \
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)) {      \
                sum = ORT_MATH_CLAMP(sum, INT32, int32_t);       \
            } else if ((onnx_type ==                             \
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)) {     \
                sum = ORT_MATH_CLAMP_MAX(sum, UINT32, uint32_t); \
            }                                                    \
            res[j] = sum;                                        \
        }                                                        \
    }

ORT_MATH_FOREACH_INTEGER_TYPE(
    ORT_MATH_MATMUL_IMPL_FOR_INTEGER_TYPE)
ORT_MATH_FOREACH_REAL_TYPE(
    ORT_MATH_MATMUL_IMPL_FOR_REAL_TYPE)
#undef ORT_MATH_MATMUL_IMPL_FOR_INTEGER_TYPE
#undef ORT_MATH_MATMUL_IMPL_FOR_REAL_TYPE

ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(
    ort_math_kernel_matmul_t, matmul)

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
    // batch_count is the number of independent matrices (outer product of batch dims)
    size_t batch_count = 1;
    for (size_t i = 0; i < batch_dims; i++) {
        if (matrix_a->shape[i] != matrix_b->shape[i]) {
            php_ort_status_throw(php_ort_status_math_error_ce,
                "matmul: batch dimension %zu mismatch (%zd vs %zd)",
                i, matrix_a->shape[i], matrix_b->shape[i]);
            return NULL;
        }
        batch_count *= matrix_a->shape[i];
    }

    ort_math_promotion_t promotion = ort_math_promotion_perform_binary(
        &ort_math_promotion_schema_matmul, matrix_a, matrix_b);
    ort_math_kernel_matmul_t kernel =
        (ort_math_kernel_matmul_t)
            ort_math_frontend_dispatch_matmul(
                &promotion,
                &ort_math_promotion_schema_matmul);
    ORT_MATH_RESULT_KERNEL_CHECK(matmul, kernel,
        &promotion, &ort_math_promotion_schema_matmul);
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

    // Prepare pointers for each batch, with type casting if needed (cast all matrices up front)
    size_t matrix_size_a = a_rows * a_cols;
    size_t matrix_size_b = b_rows * b_cols;
    size_t matrix_size_result = a_rows * b_cols;
    size_t type_size = php_ort_type_sizeof(promoted_type);
    int need_cast_a = (matrix_a->type != promoted_type);
    int need_cast_b = (matrix_b->type != promoted_type);

    void* a_buf = (void*)matrix_a->data;
    void* b_buf = (void*)matrix_b->data;
    void* tmp_a = NULL;
    void* tmp_b = NULL;

    if (need_cast_a) {
        tmp_a = ort_alloc(batch_count, matrix_size_a * type_size);
        for (size_t batch = 0; batch < batch_count; ++batch) {
            const void* src =
                (const char*)matrix_a->data +
                    batch * matrix_size_a * php_ort_type_sizeof(matrix_a->type);
            void* dst =
                (char*)tmp_a +
                    batch * matrix_size_a * type_size;

            ort_math_cast_buffer(
                src, dst, 
                matrix_a->type, promoted_type, matrix_size_a);
        }
        a_buf = tmp_a;
    }

    if (need_cast_b) {
        tmp_b = ort_alloc(batch_count, matrix_size_b * type_size);
        for (size_t batch = 0; batch < batch_count; ++batch) {
            const void* src = 
                (const char*)matrix_b->data +
                    batch * matrix_size_b * php_ort_type_sizeof(matrix_b->type);
            void* dst =
                (char*)tmp_b +
                    batch * matrix_size_b * type_size;
            ort_math_cast_buffer(
                src, dst, 
                matrix_b->type, promoted_type, matrix_size_b);
        }
        b_buf = tmp_b;
    }

    size_t chunk;
    size_t num_chunks = ort_pool_chunk(
        a_rows, type_size * b_cols, &chunk);

    ort_pool_matmul_ctx_t ctx = {
        .layout = {
            .element = type_size,
            .total = a_rows,
            .chunk = chunk
        },
        .result = result->data,
        .a = a_buf,
        .b = b_buf,
        .a_cols = a_cols,
        .b_cols = b_cols,
        .type_size = type_size,
        .matrix_size_a = matrix_size_a,
        .matrix_size_b = matrix_size_b,
        .matrix_size_result = matrix_size_result,
        .op = kernel
    };

    ort_pool_submit(ort_pool_matmul_worker, &ctx, num_chunks);

    if (tmp_a)
        ort_free(tmp_a);
    if (tmp_b)
        ort_free(tmp_b);

    return result;
}