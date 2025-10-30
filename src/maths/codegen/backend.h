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

#define ORT_MATH_BACKEND_UTIL(backend, util, generic, scalar)                \
        ort_math_backend_##backend##_##util##_##generic##_to_##scalar
#define ORT_MATH_BACKEND_UTIL_DECL(backend, util, generic, scalar, specific) \
    static zend_always_inline scalar                                \
        ORT_MATH_BACKEND_UTIL(backend, util, generic, scalar)(specific v)
#define ORT_MATH_BACKEND_UTIL_DECL_VAR(backend, util, generic, scalar, specific) \
    static zend_always_inline scalar                                    \
        ORT_MATH_BACKEND_UTIL(backend, util, generic, scalar)(specific v, size_t vl)
#define ORT_MATH_BACKEND_OP_SYMBOL(backend, op, type) \
    ort_math_backend_##backend##_##op##_##type
#define ORT_MATH_BACKEND_OP_AXIS_SYMBOL(backend, op, type) \
    ort_math_backend_##backend##_##op##_axis_##type
#define ORT_MATH_BACKEND_BINARY_OP_DECL(backend, op, type)         \
    void ORT_MATH_BACKEND_OP_SYMBOL(backend, op, type) (           \
        void* result, const void* a, const void* b, size_t count)
#define ORT_MATH_BACKEND_UNARY_OP_DECL(backend, op, type)          \
    void ORT_MATH_BACKEND_OP_SYMBOL(backend, op, type) (           \
        void* result, const void* a, size_t count)
#define ORT_MATH_BACKEND_MATMUL_OP_DECL(backend, type)             \
    void ORT_MATH_BACKEND_OP_SYMBOL(backend, matmul, type) (       \
        void* result, const void* a, const void* b,                \
        size_t a_cols, size_t b_cols)
#define ORT_MATH_BACKEND_REDUCTION_OP_DECL(backend, op, type) \
    ORT_MATH_BACKEND_UNARY_OP_DECL(backend, op, type)
#define ORT_MATH_BACKEND_REDUCTION_AXIS_OP_DECL(backend, op, type) \
    void ORT_MATH_BACKEND_OP_AXIS_SYMBOL(backend, op, type) (      \
        void* result, const void* a,                               \
        size_t outer, size_t axis, size_t inner)
#define ORT_MATH_BACKEND_SERIAL_REDUCTION_AXIS_OP_DECL(backend, op, type) \
    void ORT_MATH_BACKEND_OP_AXIS_SYMBOL(backend, serial_##op, type) (    \
        void* result, const void* a,                                      \
        size_t outer, size_t axis, size_t inner)
#endif