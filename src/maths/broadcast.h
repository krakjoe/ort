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

#ifndef HAVE_ORT_MATHS_BROADCAST
#define HAVE_ORT_MATHS_BROADCAST

#include <php.h>

#include "tensor.h"

/* Broadcasting information */
typedef struct _ort_math_broadcast_info_t {
    size_t result_dimensions;
    zend_bool is_compatible;
    zend_bool needs_broadcast_a;
    zend_bool needs_broadcast_b;
    int64_t* result_shape;
} ort_math_broadcast_info_t;

/* Broadcasting functions */
ort_math_broadcast_info_t* ort_math_broadcast_calculate(
    const int64_t* shape_a, size_t dims_a,
    const int64_t* shape_b, size_t dims_b
);

void* ort_math_broadcast_perform(
    const ort_tensor_t* result,
    const ort_tensor_t* input,
    void* out_buf
);

void ort_math_broadcast_free(ort_math_broadcast_info_t* info);
#endif