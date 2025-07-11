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

#ifndef HAVE_ORT_MATHS_BACKEND_IMPL
#define HAVE_ORT_MATHS_BACKEND_IMPL

#include <php.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "alloc.h"
#include "maths/codegen.h"
#include "maths/dispatch.h"
#include "maths/frontend/impl.h"

/** 
 * Returns largest multiple of width that is less than or equal to count.
 * 
 * Note: This function must return zero where executing subsequent operations
 *       would cause a buffer overrun or access out of bounds, or other undefined behavior.
 **/
static zend_always_inline size_t ort_math_backend_optimal_count(size_t count, size_t width) {
    if (count < width) {
        return 0; /* Not enough elements for a single operation */
    }

    return (count / width) * width;
}

/* {{{ Backend Forward Declarations of Addition Operations */
#define ORT_MATH_BACKEND_ADD_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_BINARY_OP_DECL(add, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_ADD_OP_DECL)
#undef ORT_MATH_BACKEND_ADD_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Subtraction Operations */
#define ORT_MATH_BACKEND_SUB_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_BINARY_OP_DECL(sub, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_SUB_OP_DECL)
#undef ORT_MATH_BACKEND_SUB_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Multiplication Operations */
#define ORT_MATH_BACKEND_MUL_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_BINARY_OP_DECL(mul, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_MUL_OP_DECL)
#undef ORT_MATH_BACKEND_MUL_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Division Operations */
#define ORT_MATH_BACKEND_DIV_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_BINARY_OP_DECL(div, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_DIV_OP_DECL)
#undef ORT_MATH_BACKEND_DIV_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Square Root Operations */
#define ORT_MATH_BACKEND_SQRT_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_UNARY_OP_DECL(sqrt, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_SQRT_OP_DECL)
#undef ORT_MATH_BACKEND_SQRT_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Negation Operations */
#define ORT_MATH_BACKEND_NEG_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_UNARY_OP_DECL(neg, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_NEG_OP_DECL)
#undef ORT_MATH_BACKEND_NEG_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Ceil Operations */
#define ORT_MATH_BACKEND_CEIL_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_UNARY_OP_DECL(ceil, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_CEIL_OP_DECL)
#undef ORT_MATH_BACKEND_CEIL_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Floor Operations */
#define ORT_MATH_BACKEND_FLOOR_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_UNARY_OP_DECL(floor, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_FLOOR_OP_DECL)
#undef ORT_MATH_BACKEND_FLOOR_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Round Operations */
#define ORT_MATH_BACKEND_ROUND_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_UNARY_OP_DECL(round, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_ROUND_OP_DECL)
#undef ORT_MATH_BACKEND_ROUND_OP_DECL  /* }}} */

/* {{{ Backend Forward Declarations of Truncation Operations */
#define ORT_MATH_BACKEND_TRUNC_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_UNARY_OP_DECL(trunc, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_TRUNC_OP_DECL)
#undef ORT_MATH_BACKEND_TRUNC_OP_DECL  /* }}} */

/* {{{ Backend Forward Declarations of Absolute Operations */
#define ORT_MATH_BACKEND_ABS_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_UNARY_OP_DECL(abs, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_ABS_OP_DECL)
#undef ORT_MATH_BACKEND_ABS_OP_DECL  /* }}} */

/* {{{ Backend Forward Declarations of Sign Operations */
#define ORT_MATH_BACKEND_SIGN_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_UNARY_OP_DECL(sign, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_SIGN_OP_DECL)
#undef ORT_MATH_BACKEND_SIGN_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Reciprocal Operations */
#define ORT_MATH_BACKEND_RECIP_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_UNARY_OP_DECL(recip, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_RECIP_OP_DECL)
#undef ORT_MATH_BACKEND_RECIP_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Matrix Multiplication Operations */
#define _ORT_MATH_BACKEND_MATMUL_OP_DECL(type, unused) \
        ORT_MATH_BACKEND_MATMUL_OP_DECL(type);
ORT_MATH_FOREACH_ALL_TYPES(_ORT_MATH_BACKEND_MATMUL_OP_DECL)
#undef _ORT_MATH_BACKEND_MATMUL_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Min Reduction Operations */
#define ORT_MATH_BACKEND_MIN_OP_DECL(type, unused)         \
        ORT_MATH_BACKEND_REDUCTION_OP_DECL(min, type);     \
        ORT_MATH_BACKEND_REDUCTION_AXIS_OP_DECL(min, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_MIN_OP_DECL)
#undef ORT_MATH_BACKEND_MIN_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Max Reduction Operations */
#define ORT_MATH_BACKEND_MAX_OP_DECL(type, unused)         \
        ORT_MATH_BACKEND_REDUCTION_OP_DECL(max, type);     \
        ORT_MATH_BACKEND_REDUCTION_AXIS_OP_DECL(max, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_MAX_OP_DECL)
#undef ORT_MATH_BACKEND_MAX_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Mean Reduction Operations */
#define ORT_MATH_BACKEND_MEAN_OP_DECL(type, unused)         \
        ORT_MATH_BACKEND_REDUCTION_OP_DECL(mean, type);     \
        ORT_MATH_BACKEND_REDUCTION_AXIS_OP_DECL(mean, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_MEAN_OP_DECL)
#undef ORT_MATH_BACKEND_MEAN_OP_DECL /* }}} */

/* {{{ Backend Forward Declarations of Sum Reduction Operations */
#define ORT_MATH_BACKEND_SUM_OP_DECL(type, unused)         \
        ORT_MATH_BACKEND_REDUCTION_OP_DECL(sum, type);     \
        ORT_MATH_BACKEND_REDUCTION_AXIS_OP_DECL(sum, type);
ORT_MATH_FOREACH_ALL_TYPES(ORT_MATH_BACKEND_SUM_OP_DECL)
#undef ORT_MATH_BACKEND_SUM_OP_DECL /* }}} */

/* {{{ Each backend must implement this function in its own impl.c */
void ort_math_backend_install(ort_math_dispatch_t* table); /* }}} */
#endif