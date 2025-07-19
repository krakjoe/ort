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

/**
 * @file codegen/backend.h
 * @brief Backend code generation macros
 */

#ifndef HAVE_ORT_MATHS_CODEGEN_BACKEND
#define HAVE_ORT_MATHS_CODEGEN_BACKEND

#include "maths/codegen/common.h"

#define ORT_MATH_BACKEND_OP_SYMBOL(op, type) \
    ort_math_backend_##op##_##type
#define ORT_MATH_BACKEND_OP_AXIS_SYMBOL(op, type) \
    ort_math_backend_##op##_axis_##type
#define ORT_MATH_BACKEND_BINARY_OP_DECL(op, type)                  \
    void ORT_MATH_BACKEND_OP_SYMBOL(op, type) (                    \
        void* result, const void* a, const void* b, size_t count)
#define ORT_MATH_BACKEND_UNARY_OP_DECL(op, type)                   \
    void ORT_MATH_BACKEND_OP_SYMBOL(op, type) (                    \
        void* result, const void* a, size_t count)
#define ORT_MATH_BACKEND_MATMUL_OP_DECL(type)                      \
    void ORT_MATH_BACKEND_OP_SYMBOL(matmul, type) (                \
        void* result, const void* a, const void* b,                \
        size_t a_cols, size_t b_cols)
#define ORT_MATH_BACKEND_REDUCTION_OP_DECL ORT_MATH_BACKEND_UNARY_OP_DECL
#define ORT_MATH_BACKEND_REDUCTION_AXIS_OP_DECL(op, type)          \
    void ORT_MATH_BACKEND_OP_AXIS_SYMBOL(op, type) (               \
        void* result, const void* a,                               \
        size_t outer, size_t axis, size_t inner)
#define ORT_MATH_BACKEND_SERIAL_REDUCTION_AXIS_OP_DECL(op, type)   \
    void ORT_MATH_BACKEND_OP_AXIS_SYMBOL(serial_##op, type) (      \
        void* result, const void* a,                               \
        size_t outer, size_t axis, size_t inner)

#endif