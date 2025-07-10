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

#ifndef HAVE_ORT_MATHS_FRONTEND_IMPL
#define HAVE_ORT_MATHS_FRONTEND_IMPL

/* @file impl.h
 * @brief Frontend interface for mathematical operations
 *
 * All of the functions in this file must be implemented by the frontend.
 * 
 * The implementation of these functions should be used as fallback
 * for the backend, which are defined in the backend directory.
 *
 * Note: This file is intended for use at the maths layer, and is not
 *       intended for use at the PHP layer.
 */
#include <php.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/frontend/impl.h"

/* {{{ Frontend Forward Declarations of Addition Operations */
#define ORT_MATH_FRONTEND_ADD_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_BINARY_OP_DECL(add, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_FRONTEND_ADD_OP_DECL)
#undef ORT_MATH_FRONTEND_ADD_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Subtraction Operations */
#define ORT_MATH_FRONTEND_SUB_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_BINARY_OP_DECL(sub, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_FRONTEND_SUB_OP_DECL)
#undef ORT_MATH_FRONTEND_SUB_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Multiplication Operations */
#define ORT_MATH_FRONTEND_MUL_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_BINARY_OP_DECL(mul, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_FRONTEND_MUL_OP_DECL)
#undef ORT_MATH_FRONTEND_MUL_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Division Operations */
#define ORT_MATH_FRONTEND_DIV_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_BINARY_OP_DECL(div, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_FRONTEND_DIV_OP_DECL)
#undef ORT_MATH_FRONTEND_DIV_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Square Root Operations */
#define ORT_MATH_FRONTEND_SQRT_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_UNARY_OP_DECL(sqrt, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_FRONTEND_SQRT_OP_DECL)
#undef ORT_MATH_FRONTEND_SQRT_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Ceiling Operations */
#define ORT_MATH_FRONTEND_CEIL_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_UNARY_OP_DECL(ceil, type);
ORT_MATH_FOREACH_REAL_TYPE(ORT_MATH_FRONTEND_CEIL_OP_DECL)
#undef ORT_MATH_FRONTEND_CEIL_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Floor Operations */
#define ORT_MATH_FRONTEND_FLOOR_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_UNARY_OP_DECL(floor, type);
ORT_MATH_FOREACH_REAL_TYPE(ORT_MATH_FRONTEND_FLOOR_OP_DECL)
#undef ORT_MATH_FRONTEND_FLOOR_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Round Operations */
#define ORT_MATH_FRONTEND_ROUND_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_UNARY_OP_DECL(round, type);
ORT_MATH_FOREACH_REAL_TYPE(ORT_MATH_FRONTEND_ROUND_OP_DECL)
#undef ORT_MATH_FRONTEND_ROUND_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Truncation Operations */
#define ORT_MATH_FRONTEND_TRUNC_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_UNARY_OP_DECL(trunc, type);
ORT_MATH_FOREACH_REAL_TYPE(ORT_MATH_FRONTEND_TRUNC_OP_DECL)
#undef ORT_MATH_FRONTEND_TRUNC_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Absolute Operations */
#define ORT_MATH_FRONTEND_ABS_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_UNARY_OP_DECL(abs, type);
ORT_MATH_FOREACH_REAL_TYPE(ORT_MATH_FRONTEND_ABS_OP_DECL)
#undef ORT_MATH_FRONTEND_ABS_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Negation Operations */
#define ORT_MATH_FRONTEND_NEG_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_UNARY_OP_DECL(neg, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_FRONTEND_NEG_OP_DECL)
#undef ORT_MATH_FRONTEND_NEG_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Reciprocal Operations */
#define ORT_MATH_FRONTEND_RECIP_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_UNARY_OP_DECL(recip, type);
ORT_MATH_FOREACH_REAL_TYPE(ORT_MATH_FRONTEND_RECIP_OP_DECL)
#undef ORT_MATH_FRONTEND_RECIP_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Sign Operations */
#define ORT_MATH_FRONTEND_SIGN_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_UNARY_OP_DECL(sign, type);
ORT_MATH_FOREACH_REAL_TYPE(ORT_MATH_FRONTEND_SIGN_OP_DECL)
ORT_MATH_FOREACH_SIGNED_TYPE(ORT_MATH_FRONTEND_SIGN_OP_DECL)
#undef ORT_MATH_FRONTEND_SIGN_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Modulo Operations */
#define ORT_MATH_FRONTEND_MOD_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_BINARY_OP_DECL(mod, type);
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_FRONTEND_MOD_OP_DECL)
#undef ORT_MATH_FRONTEND_MOD_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Power Operations */
#define ORT_MATH_FRONTEND_POW_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_BINARY_OP_DECL(pow, type);
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_FRONTEND_POW_OP_DECL)
#undef ORT_MATH_FRONTEND_POW_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Dot Product Operations */
#define ORT_MATH_FRONTEND_DOT_OP_DECL(type, unused) \
      ORT_MATH_FRONTEND_BINARY_OP_DECL(dot, type);
ORT_MATH_FOREACH_NUMERIC_TYPE(ORT_MATH_FRONTEND_DOT_OP_DECL)
#undef ORT_MATH_FRONTEND_DOT_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Matrix Multiplication Operations */
#define _ORT_MATH_FRONTEND_MATMUL_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_MATMUL_OP_DECL(type);
ORT_MATH_FOREACH_NUMERIC_TYPE(_ORT_MATH_FRONTEND_MATMUL_OP_DECL)
#undef ORT_MATH_FRONTEND_MATMUL_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Minimum Operations */
#define ORT_MATH_FRONTEND_MIN_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_REDUCTION_OP_DECL(min, type); \
        ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(min, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_FRONTEND_MIN_OP_DECL)
#undef ORT_MATH_FRONTEND_MIN_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Maximum Operations */
#define _ORT_MATH_FRONTEND_MAX_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_REDUCTION_OP_DECL(max, type); \
        ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(max, type);
ORT_MATH_FOREACH_ALL_TYPES(_ORT_MATH_FRONTEND_MAX_OP_DECL)
#undef _ORT_MATH_FRONTEND_MAX_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Softmax Operations */
#define ORT_MATH_FRONTEND_SOFTMAX_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(softmax, type);
ORT_MATH_FOREACH_REAL_TYPE(ORT_MATH_FRONTEND_SOFTMAX_OP_DECL)
#undef ORT_MATH_FRONTEND_SOFTMAX_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Mean Operations */
#define ORT_MATH_FRONTEND_MEAN_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_REDUCTION_OP_DECL(mean, type); \
        ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(mean, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_FRONTEND_MEAN_OP_DECL)
#undef ORT_MATH_FRONTEND_MEAN_OP_DECL /* }}} */

/* {{{ Frontend Forward Declarations of Sum Operations */
#define ORT_MATH_FRONTEND_SUM_OP_DECL(type, unused) \
        ORT_MATH_FRONTEND_REDUCTION_OP_DECL(sum, type); \
        ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(sum, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_FRONTEND_SUM_OP_DECL)
#undef ORT_MATH_FRONTEND_SUM_OP_DECL /* }}} */

#endif