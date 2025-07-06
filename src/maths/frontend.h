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

#ifndef HAVE_ORT_MATHS_FRONTEND
#define HAVE_ORT_MATHS_FRONTEND

/* @file frontend.h
 * @brief Frontend interface for mathematical operations
 *
 * All of the functions in this file must be implemented by the frontend.
 * 
 * The implementation of these functions should be used as fallback
 * for the SIMD backends, which are defined in the simd directory.
 *
 * Note: This file is intended for use at the maths layer, and is not
 *       intended for use at the PHP layer.
 */
#include "ort.h"

/* Mathematical operation implementations */
extern void ort_math_ops_add_float(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_double(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_int8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_int16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_int32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_int64_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_uint8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_uint16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_uint32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_add_zend_bool(void* result, const void* a, const void* b, size_t count);

extern void ort_math_ops_sub_float(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_double(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_int8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_int16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_int32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_int64_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_uint8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_uint16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_uint32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_sub_zend_bool(void* result, const void* a, const void* b, size_t count);

extern void ort_math_ops_mul_float(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_double(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_int8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_int16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_int32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_int64_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_uint8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_uint16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_uint32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mul_zend_bool(void* result, const void* a, const void* b, size_t count);

extern void ort_math_ops_div_float(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_double(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_int8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_int16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_int32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_int64_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_uint8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_uint16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_uint32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_div_zend_bool(void* result, const void* a, const void* b, size_t count);

extern void ort_math_ops_sqrt_int8_t(void* result, const void* a, size_t count);
extern void ort_math_ops_sqrt_int16_t(void* result, const void* a, size_t count);
extern void ort_math_ops_sqrt_int32_t(void* result, const void* a, size_t count);
extern void ort_math_ops_sqrt_int64_t(void* result, const void* a, size_t count);
extern void ort_math_ops_sqrt_uint8_t(void* result, const void* a, size_t count);
extern void ort_math_ops_sqrt_uint16_t(void* result, const void* a, size_t count);
extern void ort_math_ops_sqrt_uint32_t(void* result, const void* a, size_t count);
extern void ort_math_ops_sqrt_float(void* result, const void* a, size_t count);
extern void ort_math_ops_sqrt_double(void* result, const void* a, size_t count);

extern void ort_math_ops_ceil_float(void* result, const void* a, size_t count);
extern void ort_math_ops_ceil_double(void* result, const void* a, size_t count);

extern void ort_math_ops_floor_float(void* result, const void* a, size_t count);
extern void ort_math_ops_floor_double(void* result, const void* a, size_t count);

extern void ort_math_ops_round_float(void* result, const void* a, size_t count);
extern void ort_math_ops_round_double(void* result, const void* a, size_t count);

extern void ort_math_ops_trunc_float(void* result, const void* a, size_t count);
extern void ort_math_ops_trunc_double(void* result, const void* a, size_t count);

extern void ort_math_ops_abs_float(void* result, const void* a, size_t count);
extern void ort_math_ops_abs_double(void* result, const void* a, size_t count);

extern void ort_math_ops_sign_float(void* result, const void* a, size_t count);
extern void ort_math_ops_sign_double(void* result, const void* a, size_t count);

extern void ort_math_ops_recip_float(void* result, const void* a, size_t count);
extern void ort_math_ops_recip_double(void* result, const void* a, size_t count);

extern void ort_math_ops_neg_float(void* result, const void* a, size_t count);
extern void ort_math_ops_neg_double(void* result, const void* a, size_t count);

extern void ort_math_ops_mod_float(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_double(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_int8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_int16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_int32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_int64_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_uint8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_uint16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_mod_uint32_t(void* result, const void* a, const void* b, size_t count);

extern void ort_math_ops_pow_int8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_pow_int16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_pow_int32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_pow_int64_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_pow_uint8_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_pow_uint16_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_pow_uint32_t(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_pow_float(void* result, const void* a, const void* b, size_t count);
extern void ort_math_ops_pow_double(void* result, const void* a, const void* b, size_t count);
#endif