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

#include "tensor.h"

#include "maths/cast.h"
#include "maths/broadcast.h"
#include "maths/result.h"

/* Broadcasting implementation */
ort_math_broadcast_info_t* ort_math_broadcast_calculate(
    const int64_t* shape_a, size_t dims_a,
    const int64_t* shape_b, size_t dims_b
) {
    ort_math_broadcast_info_t* info = ecalloc(1, 
        sizeof(ort_math_broadcast_info_t) +
            (sizeof(int64_t) * MAX(dims_a, dims_b)));

    /* Determine result dimensions (max of both) */
    info->result_dimensions = MAX(dims_a, dims_b);
    info->is_compatible = 1;
    info->needs_broadcast_a = 0;
    info->needs_broadcast_b = 0;
    info->result_shape = (int64_t*)(info + 1);

    /* Work backwards from the last dimension */
    for (size_t i = 0; i < info->result_dimensions; i++) {
        size_t idx_a = dims_a > i ? dims_a - 1 - i : SIZE_MAX;
        size_t idx_b = dims_b > i ? dims_b - 1 - i : SIZE_MAX;
        size_t result_idx = info->result_dimensions - 1 - i;
        
        int64_t dim_a = (idx_a != SIZE_MAX) ? shape_a[idx_a] : 1;
        int64_t dim_b = (idx_b != SIZE_MAX) ? shape_b[idx_b] : 1;
        
        if (dim_a == dim_b) {
            info->result_shape[result_idx] = dim_a;
        } else if (dim_a == 1) {
            info->result_shape[result_idx] = dim_b;
            info->needs_broadcast_a = 1;
        } else if (dim_b == 1) {
            info->result_shape[result_idx] = dim_a; 
            info->needs_broadcast_b = 1;
        } else {
            /* Incompatible shapes */
            info->is_compatible = 0;
            break;
        }
    }
    
    return info;
}

#define ORT_MATH_RESULT_STACK_DIMENSIONS 8

// Centralized upcast + broadcast routine for elementwise ops
void* ort_math_broadcast_perform(
    const ort_tensor_t* result,
    const ort_tensor_t* input,
    void* out_buf
) {
    // Broadcast input to result shape and upcast to result type
    int64_t out_indices[ORT_MATH_RESULT_STACK_DIMENSIONS];
    int64_t in_indices[ORT_MATH_RESULT_STACK_DIMENSIONS];

    size_t in_offset = result->dimensions - input->dimensions;
    size_t in_type_size =
        php_ort_type_sizeof(input->type);
    size_t out_type_size =
        php_ort_type_sizeof(result->type);

    for (size_t i = 0; i < result->elements; ++i) {
        ort_math_result_multi(
            i, 
            result->shape, 
            result->dimensions,
            out_indices);

        for (size_t d = 0; d < result->dimensions; ++d) {
            in_indices[d] = (d < in_offset) ? 0 :
                (input->shape[d - in_offset] == 1 ?
                    0 : out_indices[d]);
        }

        zend_long in_flat =
            ort_math_result_flat(
                in_indices + in_offset,
                input->shape, input->dimensions);

        void* src_ptr =
            (char*)input->data +
                in_flat * in_type_size;
        void* dst_ptr =
            (char*)out_buf +
                i * out_type_size;

        ort_math_cast_element(
            src_ptr,
            dst_ptr,
            input->type,
            result->type);
    }

    return out_buf;
}

void ort_math_broadcast_free(ort_math_broadcast_info_t* info) {
    if (info) {
        efree(info);
    }
}
