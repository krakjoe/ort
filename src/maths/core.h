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

#ifndef HAVE_ORT_MATHS_CORE
#define HAVE_ORT_MATHS_CORE

/* @file core.h
 * @brief Core math API for implementation of mathematical operations.
 * 
 * Note: This file is intended for use at the C layer, and is not
 *       intended for use at the PHP layer.
 */
#include "ort.h"

/** {{{  Startup and shutdown, called from the PHP layer */
void ort_math_startup(void);
void ort_math_shutdown(void); /* }}} */

/** {{{  Activate and deactivate, called from the PHP layer */
void ort_math_activate(void);
void ort_math_deactivate(void); /* }}} */

/* Core mathematical operation function pointers */
typedef void (*ort_math_kernel_binary_t)(
  void* result, const void* a, const void* b, size_t count);
typedef void (*ort_math_kernel_scalar_t)(
  void* result, const void* a, const void* b, size_t count);
typedef void (*ort_math_kernel_unary_t)(
  void* result, const void* input, size_t count);
typedef void (*ort_math_kernel_matmul_t)(
  void* result, const void* a, const void* b, size_t a_cols, size_t b_cols);

typedef ort_math_kernel_unary_t ort_math_kernel_reduce_tensor_t;
typedef void (*ort_math_kernel_reduce_axis_t)(
    void *result,
    const void *a,
    const int64_t *input_shape, size_t input_dims,
    const int64_t *output_shape, size_t output_dims,
    size_t axis);

/* {{{ Clamping is required everywhere */
#define ORT_MATH_CLAMP(val, range, type)  \
    (val > range##_MAX) ? range##_MAX :   \
    (val < range##_MIN) ? range##_MIN : val /* }}} */
/* {{{ Clamping at max is required everywhere min is undefined */
#define ORT_MATH_CLAMP_MAX(val, range, type) \
    (val > range##_MAX) ? range##_MAX : val /* }}} */
#endif
