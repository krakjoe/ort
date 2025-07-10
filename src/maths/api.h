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

#ifndef HAVE_ORT_MATHS_API
#define HAVE_ORT_MATHS_API

/** {{{
  * @file api.h
  * @brief Math operations API for the ORT extension.
  *
  * This file declares the functions for performing various mathematical
  * operations on tensors, including element-wise operations, scalar
  * operations, trigonometric functions, hyperbolic functions, exponential
  * and logarithmic functions, unary operations, linear algebra operations,
  * and reduction operations.
  * 
  * Note: This is intended for inclusion at the PHP layer, and is not
  *       intended for use at the C layer.
}}} */
#include <php.h>

#include "maths/result.h"

/* Element-wise operations */
ort_tensor_t* ort_math_result_add(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_tensor_t* ort_math_result_subtract(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_tensor_t* ort_math_result_multiply(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_tensor_t* ort_math_result_divide(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_tensor_t* ort_math_result_pow(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_tensor_t* ort_math_result_mod(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);

/* Scalar operations */
ort_tensor_t* ort_math_result_add_scalar(ort_tensor_t* tensor, zval* scalar);
ort_tensor_t* ort_math_result_subtract_scalar(ort_tensor_t* tensor, zval* scalar);
ort_tensor_t* ort_math_result_multiply_scalar(ort_tensor_t* tensor, zval* scalar);
ort_tensor_t* ort_math_result_divide_scalar(ort_tensor_t* tensor, zval* scalar);
ort_tensor_t* ort_math_result_pow_scalar(ort_tensor_t* tensor, zval* scalar);
ort_tensor_t* ort_math_result_mod_scalar(ort_tensor_t* tensor, zval* scalar);

/* Trigonometric operations */
ort_tensor_t* ort_math_result_sin(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_cos(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_tan(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_asin(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_acos(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_atan(ort_tensor_t* tensor);

/* Hyperbolic operations */
ort_tensor_t* ort_math_result_sinh(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_cosh(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_tanh(ort_tensor_t* tensor);

/* Exponential and logarithmic operations */
ort_tensor_t* ort_math_result_exp(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_exp2(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_log(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_log2(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_log10(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_sqrt(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_cbrt(ort_tensor_t* tensor);

/* Element-wise unary operations */
ort_tensor_t* ort_math_result_abs(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_sign(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_recip(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_ceil(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_floor(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_round(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_trunc(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_neg(ort_tensor_t* tensor);

/* Linear algebra operations */
ort_tensor_t* ort_math_result_matmul(ort_tensor_t* matrix_a, ort_tensor_t* matrix_b);

/* {{{ TODO(krakjoe) */
ort_tensor_t* ort_math_result_transpose(ort_tensor_t* matrix);
ort_tensor_t* ort_math_result_inverse(ort_tensor_t* matrix); /* }}} */

/* Reduction operations */
ort_tensor_t* ort_math_result_dot(ort_tensor_t* vector_a, ort_tensor_t* vector_b);
ort_tensor_t* ort_math_result_sum(ort_tensor_t* tensor, zval* axis, zend_bool keepdims);
ort_tensor_t* ort_math_result_mean(ort_tensor_t* tensor, zval* axis, zend_bool keepdims);
ort_tensor_t* ort_math_result_min(ort_tensor_t* tensor, zval* axis, zend_bool keepdims);
ort_tensor_t* ort_math_result_max(ort_tensor_t* tensor, zval* axis, zend_bool keepdims);
ort_tensor_t* ort_math_result_softmax(ort_tensor_t* tensor, zval* axis, zend_bool keepdims);

/* {{{ TODO(krakjoe) */
ort_tensor_t* ort_math_result_std(ort_tensor_t* tensor, zval* axis, zend_long ddof, zend_bool keepdims);
ort_tensor_t* ort_math_result_var(ort_tensor_t* tensor, zval* axis, zend_long ddof, zend_bool keepdims);
/* Shape operations */
ort_tensor_t* ort_math_result_reshape(ort_tensor_t* tensor, zval* new_shape);
ort_tensor_t* ort_math_result_flatten(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_squeeze(ort_tensor_t* tensor, zval* axis);
ort_tensor_t* ort_math_result_unsqueeze(ort_tensor_t* tensor, zend_long axis);
/* }}} */

/** {{{ Reduction operations */
ort_tensor_t* ort_math_result_reduce_tensor_min(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_reduce_axis_min(
  ort_tensor_t* tensor, zend_long axis, zend_bool keepdims);
ort_tensor_t* ort_math_result_reduce_tensor_max(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_reduce_axis_max(
  ort_tensor_t* tensor, zend_long axis, zend_bool keepdims);
ort_tensor_t* ort_math_result_reduce_tensor_mean(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_reduce_axis_mean(
  ort_tensor_t* tensor, zend_long axis, zend_bool keepdims);
ort_tensor_t* ort_math_result_reduce_tensor_sum(ort_tensor_t* tensor);
ort_tensor_t* ort_math_result_reduce_axis_sum(
  ort_tensor_t* tensor, zend_long axis, zend_bool keepdims);
ort_tensor_t* ort_math_result_reduce_axis_softmax(
  ort_tensor_t* tensor, zend_long axis, zend_bool keepdims);
/* }}} */
#endif