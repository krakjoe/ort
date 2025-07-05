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

#ifndef HAVE_ORT_MATHS_OPS
#define HAVE_ORT_MATHS_OPS

#include <php.h>
#include "tensor.h"
#include "maths/core.h"

/* Element-wise arithmetic operations */
ort_math_result_t* ort_math_result_add(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_math_result_t* ort_math_result_subtract(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_math_result_t* ort_math_result_multiply(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_math_result_t* ort_math_result_divide(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_math_result_t* ort_math_result_power(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_math_result_t* ort_math_result_modulo(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);

/* Scalar operations */
ort_math_result_t* ort_math_result_add_scalar(ort_tensor_t* tensor, zval* scalar);
ort_math_result_t* ort_math_result_subtract_scalar(ort_tensor_t* tensor, zval* scalar);
ort_math_result_t* ort_math_result_multiply_scalar(ort_tensor_t* tensor, zval* scalar);
ort_math_result_t* ort_math_result_divide_scalar(ort_tensor_t* tensor, zval* scalar);
ort_math_result_t* ort_math_result_power_scalar(ort_tensor_t* tensor, zval* scalar);
ort_math_result_t* ort_math_result_modulo_scalar(ort_tensor_t* tensor, zval* scalar);

/* Comparison operations */
ort_math_result_t* ort_math_result_equal(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_math_result_t* ort_math_result_not_equal(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_math_result_t* ort_math_result_greater(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_math_result_t* ort_math_result_greater_equal(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_math_result_t* ort_math_result_less(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);
ort_math_result_t* ort_math_result_less_equal(ort_tensor_t* tensor_a, ort_tensor_t* tensor_b);

/* Trigonometric operations */
ort_math_result_t* ort_math_result_sin(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_cos(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_tan(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_asin(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_acos(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_atan(ort_tensor_t* tensor);

/* Hyperbolic operations */
ort_math_result_t* ort_math_result_sinh(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_cosh(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_tanh(ort_tensor_t* tensor);

/* Exponential and logarithmic operations */
ort_math_result_t* ort_math_result_exp(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_exp2(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_log(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_log2(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_log10(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_sqrt(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_cbrt(ort_tensor_t* tensor);

/* Element-wise unary operations */
ort_math_result_t* ort_math_result_abs(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_sign(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_reciprocal(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_ceil(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_floor(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_round(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_trunc(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_negative(ort_tensor_t* tensor);

/* Linear algebra operations */
ort_math_result_t* ort_math_result_matmul(ort_tensor_t* matrix_a, ort_tensor_t* matrix_b);
ort_math_result_t* ort_math_result_dot(ort_tensor_t* vector_a, ort_tensor_t* vector_b);
ort_math_result_t* ort_math_result_transpose(ort_tensor_t* matrix);
ort_math_result_t* ort_math_result_inverse(ort_tensor_t* matrix);

/* Reduction operations */
ort_math_result_t* ort_math_result_sum(ort_tensor_t* tensor, zval* axis, zend_bool keepdims);
ort_math_result_t* ort_math_result_mean(ort_tensor_t* tensor, zval* axis, zend_bool keepdims);
ort_math_result_t* ort_math_result_std(ort_tensor_t* tensor, zval* axis, zend_long ddof, zend_bool keepdims);
ort_math_result_t* ort_math_result_var(ort_tensor_t* tensor, zval* axis, zend_long ddof, zend_bool keepdims);
ort_math_result_t* ort_math_result_min(ort_tensor_t* tensor, zval* axis, zend_bool keepdims);
ort_math_result_t* ort_math_result_max(ort_tensor_t* tensor, zval* axis, zend_bool keepdims);

/* Shape operations */
ort_math_result_t* ort_math_result_reshape(ort_tensor_t* tensor, zval* new_shape);
ort_math_result_t* ort_math_result_flatten(ort_tensor_t* tensor);
ort_math_result_t* ort_math_result_squeeze(ort_tensor_t* tensor, zval* axis);
ort_math_result_t* ort_math_result_unsqueeze(ort_tensor_t* tensor, zend_long axis);

#endif
