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

#ifndef HAVE_ORT_MATHS_VALIDATE
#define HAVE_ORT_MATHS_VALIDATE

#include "tensor.h"

zend_bool ort_math_validate_input(
    ort_tensor_t* tensor, const char* operation_name);
zend_bool ort_math_validate_axis(
    ort_tensor_t* tensor, zend_long axis, const char* operation_name);

static zend_always_inline zend_bool ort_math_validate_identity(
    ort_tensor_t* tensor_a, ort_tensor_t* tensor_b) {
    /* Type must match to be considered identical */
    if (tensor_a->type != tensor_b->type) {
        return 0;
    }

    /* Dimensions must match */
    if (tensor_a->dimensions != tensor_b->dimensions) {
        return 0;
    }

    /* Compare each dimension */
    for (size_t i = 0; i < tensor_a->dimensions; i++) {
        if (tensor_a->shape[i] != tensor_b->shape[i]) {
            return 0;
        }
    }

    return 1;
}

#endif