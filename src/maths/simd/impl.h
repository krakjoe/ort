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

#ifndef HAVE_ORT_MATHS_SIMD_IMPL
#define HAVE_ORT_MATHS_SIMD_IMPL

#include <php.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "maths/dispatch.h"
#include "maths/frontend.h"

/** 
 * Returns largest multiple of width that is less than or equal to count.
 * 
 * Note: This function must return zero where executing subsequent SIMD operations
 *       would cause a buffer overrun or access out of bounds, or other undefined behavior.
 **/
static zend_always_inline size_t ort_math_simd_optimal_count(size_t count, size_t width) {
    if (count < width) {
        return 0; /* Not enough elements for a single SIMD operation */
    }

    return (count / width) * width;
}

/** 
 * SIMD Code Generation Macros
 ***/

#define ORT_MATH_SIMD_BINARY_OP_DECL(op, type)                     \
    extern void ort_math_simd_##op##_##type(                       \
        void* result, const void* a, const void* b, size_t count)
#define ORT_MATH_SIMD_UNARY_OP_DECL(op, type)                      \
    extern void ort_math_simd_##op##_##type(                       \
        void* result, const void* a, size_t count)
#define ORT_MATH_SIMD_MATMUL_OP_DECL(type)                         \
    extern void ort_math_simd_matmul_##type(                       \
        void* result, const void* a, const void* b,                \
        size_t a_rows, size_t a_cols, size_t b_cols)

#define ORT_MATH_SIMD_REDUCTION_OP_DECL ORT_MATH_SIMD_UNARY_OP_DECL
#define ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(op, type)             \
    extern void ort_math_simd_##op##_axis_##type(                  \
        void* result, const void* a,                               \
        size_t outer_size, size_t axis_size, size_t inner_size)

/**
 * SIMD Forward Declarations of Contracted Binary Operations
 * 
 * Note: These contracts may be implemented by the SIMD backend.
 */

/* {{{ SIMD Forward Declarations of Contracted Addition Operations */
ORT_MATH_SIMD_BINARY_OP_DECL(add, int8_t);
ORT_MATH_SIMD_BINARY_OP_DECL(add, int16_t);
ORT_MATH_SIMD_BINARY_OP_DECL(add, int32_t);
ORT_MATH_SIMD_BINARY_OP_DECL(add, int64_t);

ORT_MATH_SIMD_BINARY_OP_DECL(add, uint8_t);
ORT_MATH_SIMD_BINARY_OP_DECL(add, uint16_t);
ORT_MATH_SIMD_BINARY_OP_DECL(add, uint32_t);

ORT_MATH_SIMD_BINARY_OP_DECL(add, float);
ORT_MATH_SIMD_BINARY_OP_DECL(add, double); /* }}} */

/* {{{ SIMD Forward Declarations of Contracted Subtraction Operations */
ORT_MATH_SIMD_BINARY_OP_DECL(sub, int8_t);
ORT_MATH_SIMD_BINARY_OP_DECL(sub, int16_t);
ORT_MATH_SIMD_BINARY_OP_DECL(sub, int32_t);
ORT_MATH_SIMD_BINARY_OP_DECL(sub, int64_t);

ORT_MATH_SIMD_BINARY_OP_DECL(sub, uint8_t);
ORT_MATH_SIMD_BINARY_OP_DECL(sub, uint16_t);
ORT_MATH_SIMD_BINARY_OP_DECL(sub, uint32_t);

ORT_MATH_SIMD_BINARY_OP_DECL(sub, float);
ORT_MATH_SIMD_BINARY_OP_DECL(sub, double); /* }}} */

/* {{{ SIMD Forward Declarations of Contracted Muiltiplication Operations */
ORT_MATH_SIMD_BINARY_OP_DECL(mul, int8_t);
ORT_MATH_SIMD_BINARY_OP_DECL(mul, int16_t);
ORT_MATH_SIMD_BINARY_OP_DECL(mul, int32_t);
ORT_MATH_SIMD_BINARY_OP_DECL(mul, int64_t);

ORT_MATH_SIMD_BINARY_OP_DECL(mul, uint8_t);
ORT_MATH_SIMD_BINARY_OP_DECL(mul, uint16_t);
ORT_MATH_SIMD_BINARY_OP_DECL(mul, uint32_t);

ORT_MATH_SIMD_BINARY_OP_DECL(mul, float);
ORT_MATH_SIMD_BINARY_OP_DECL(mul, double); /* }}} */

/* {{{ SIMD Forward Declarations of Contracted Division Operations */
ORT_MATH_SIMD_BINARY_OP_DECL(div, float);
ORT_MATH_SIMD_BINARY_OP_DECL(div, double); /* }}} */

/* {{{ SIMD Forward Declarations for Contracted Square Root Operations */
ORT_MATH_SIMD_UNARY_OP_DECL(sqrt, float);
ORT_MATH_SIMD_UNARY_OP_DECL(sqrt, double); /* }}} */

/* {{{ SIMD Forward Declarations for Contracted Negation Operations */
ORT_MATH_SIMD_UNARY_OP_DECL(neg, float);
ORT_MATH_SIMD_UNARY_OP_DECL(neg, double); /* }}} */

/* {{{ SIMD Forward Declarations for Contracted Ceil Operations */
ORT_MATH_SIMD_UNARY_OP_DECL(ceil, float);
ORT_MATH_SIMD_UNARY_OP_DECL(ceil, double); /* }}} */

/* {{{ SIMD Forward Declarations for Contracted Floor Operations */
ORT_MATH_SIMD_UNARY_OP_DECL(floor, float);
ORT_MATH_SIMD_UNARY_OP_DECL(floor, double); /* }}} */

/* {{{ SIMD Forward Declarations for Contracted Round Operations */
ORT_MATH_SIMD_UNARY_OP_DECL(round, float);
ORT_MATH_SIMD_UNARY_OP_DECL(round, double); /* }}} */

/* {{{ SIMD Forward Declarations for Contracted Truncation Operations */
ORT_MATH_SIMD_UNARY_OP_DECL(trunc, float);
ORT_MATH_SIMD_UNARY_OP_DECL(trunc, double); /* }}} */

/** SIMD Forward Declarations for Contracted Absolute Operations */
ORT_MATH_SIMD_UNARY_OP_DECL(abs, float);
ORT_MATH_SIMD_UNARY_OP_DECL(abs, double); /* }}} */

/* {{{ SIMD Forward Declarations for Contracted Sign Operations */
ORT_MATH_SIMD_UNARY_OP_DECL(sign, float);
ORT_MATH_SIMD_UNARY_OP_DECL(sign, double); /* }}} */

/* {{{ SIMD Forward Declarations for Contracted Reciprocal Operations */
ORT_MATH_SIMD_UNARY_OP_DECL(recip, float);
ORT_MATH_SIMD_UNARY_OP_DECL(recip, double); /* }}} */

/* {{{ SIMD Forward Declarations for Matrix Multiplication Operations */
ORT_MATH_SIMD_MATMUL_OP_DECL(float);
ORT_MATH_SIMD_MATMUL_OP_DECL(double);
ORT_MATH_SIMD_MATMUL_OP_DECL(int8_t);
ORT_MATH_SIMD_MATMUL_OP_DECL(int16_t);
ORT_MATH_SIMD_MATMUL_OP_DECL(int32_t);
ORT_MATH_SIMD_MATMUL_OP_DECL(int64_t);
ORT_MATH_SIMD_MATMUL_OP_DECL(uint8_t);
ORT_MATH_SIMD_MATMUL_OP_DECL(uint16_t);
ORT_MATH_SIMD_MATMUL_OP_DECL(uint32_t); /* }}} */

/* {{{ SIMD Forward Declarations for Min Reduction Operations */
ORT_MATH_SIMD_REDUCTION_OP_DECL(min, float);
ORT_MATH_SIMD_REDUCTION_OP_DECL(min, double);
ORT_MATH_SIMD_REDUCTION_OP_DECL(min, int8_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(min, int16_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(min, int32_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(min, int64_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(min, uint8_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(min, uint16_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(min, uint32_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(min, zend_bool);

ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(min, float);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(min, double);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(min, int8_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(min, int16_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(min, int32_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(min, int64_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(min, uint8_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(min, uint16_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(min, uint32_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(min, zend_bool); /* }}} */

/* {{{ SIMD Forward Declarations for Max Reduction Operations */
ORT_MATH_SIMD_REDUCTION_OP_DECL(max, float);
ORT_MATH_SIMD_REDUCTION_OP_DECL(max, double);
ORT_MATH_SIMD_REDUCTION_OP_DECL(max, int8_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(max, int16_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(max, int32_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(max, int64_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(max, uint8_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(max, uint16_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(max, uint32_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(max, zend_bool);

ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(max, float);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(max, double);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(max, int8_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(max, int16_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(max, int32_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(max, int64_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(max, uint8_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(max, uint16_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(max, uint32_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(max, zend_bool); /* }}} */

/* {{{ SIMD Forward Declarations for Mean Reduction Operations */
ORT_MATH_SIMD_REDUCTION_OP_DECL(mean, float);
ORT_MATH_SIMD_REDUCTION_OP_DECL(mean, double);
ORT_MATH_SIMD_REDUCTION_OP_DECL(mean, int8_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(mean, int16_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(mean, int32_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(mean, int64_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(mean, uint8_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(mean, uint16_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(mean, uint32_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(mean, zend_bool);

ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(mean, float);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(mean, double);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(mean, int8_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(mean, int16_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(mean, int32_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(mean, int64_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(mean, uint8_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(mean, uint16_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(mean, uint32_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(mean, zend_bool); /* }}} */

/* {{{ SIMD Forward Declarations for Sum Reduction Operations */
ORT_MATH_SIMD_REDUCTION_OP_DECL(sum, float);
ORT_MATH_SIMD_REDUCTION_OP_DECL(sum, double);
ORT_MATH_SIMD_REDUCTION_OP_DECL(sum, int8_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(sum, int16_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(sum, int32_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(sum, int64_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(sum, uint8_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(sum, uint16_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(sum, uint32_t);
ORT_MATH_SIMD_REDUCTION_OP_DECL(sum, zend_bool);

ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(sum, float);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(sum, double);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(sum, int8_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(sum, int16_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(sum, int32_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(sum, int64_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(sum, uint8_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(sum, uint16_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(sum, uint32_t);
ORT_MATH_SIMD_REDUCTION_AXIS_OP_DECL(sum, zend_bool); /* }}} */

/* {{{ Each backend must implement this function in its own impl.c */
void ort_math_simd_install(ort_math_dispatch_t* table); /* }}} */
#endif