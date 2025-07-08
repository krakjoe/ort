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

#include "maths/broadcast.h"

/* Broadcasting implementation */
ort_math_broadcast_info_t* ort_math_broadcast_calculate(
    const int64_t* shape_a, size_t dims_a,
    const int64_t* shape_b, size_t dims_b
) {
    ort_math_broadcast_info_t* info = ecalloc(1, sizeof(ort_math_broadcast_info_t));
    if (!info) {
        return NULL;
    }
    memset(info, 0, sizeof(ort_math_broadcast_info_t));
    
    /* Determine result dimensions (max of both) */
    info->result_dimensions = dims_a > dims_b ? dims_a : dims_b;
    info->result_shape = ecalloc(info->result_dimensions, sizeof(int64_t));
    info->is_compatible = 1;
    info->needs_broadcast_a = 0;
    info->needs_broadcast_b = 0;
    
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

void ort_math_broadcast_free(ort_math_broadcast_info_t* info) {
    if (info) {
        if (info->result_shape) {
            efree(info->result_shape);
        }
        efree(info);
    }
}
