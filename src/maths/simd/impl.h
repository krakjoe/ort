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

/* {{{ Each backend must implement this function in its own impl.c */
void ort_math_simd_install(ort_math_dispatch_t* table); /* }}} */
#endif