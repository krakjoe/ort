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
 * @file codegen/frontend.h
 * @brief Frontend code generation macros
 */

#ifndef HAVE_ORT_MATHS_CODEGEN_FRONTEND
#define HAVE_ORT_MATHS_CODEGEN_FRONTEND

#include "maths/codegen/common.h"

#define ORT_MATH_FRONTEND_OP_SYMBOL(op, type) \
    ort_math_frontend_##op##_##type
#define ORT_MATH_FRONTEND_OP_AXIS_SYMBOL(op, type) \
    ort_math_frontend_##op##_axis_##type
#define ORT_MATH_FRONTEND_BINARY_OP_DECL(op, type)                    \
    ORT_MATH_KERNEL_ALIGN                                             \
        void ORT_MATH_FRONTEND_OP_SYMBOL(op, type)(                   \
            void* result, const void* a, const void* b, size_t count)
#define ORT_MATH_FRONTEND_UNARY_OP_DECL(op, type)                     \
    ORT_MATH_KERNEL_ALIGN                                             \
        void ORT_MATH_FRONTEND_OP_SYMBOL(op, type)(                   \
            void* result, const void* a, size_t count)
#define ORT_MATH_FRONTEND_SCALAR_OP_DECL(op, type)                    \
    ORT_MATH_FRONTEND_BINARY_OP_DECL(op##_scalar, type)
#define ORT_MATH_FRONTEND_MATMUL_OP_DECL(type)                        \
    ORT_MATH_KERNEL_ALIGN                                             \
        void ORT_MATH_FRONTEND_OP_SYMBOL(matmul, type)(               \
            void* result, const void* a, const void* b,               \
            size_t a_cols, size_t b_cols)
#define ORT_MATH_FRONTEND_REDUCTION_OP_DECL ORT_MATH_FRONTEND_UNARY_OP_DECL
#define ORT_MATH_FRONTEND_REDUCTION_AXIS_OP_DECL(op, type)            \
    ORT_MATH_KERNEL_ALIGN                                             \
        void ORT_MATH_FRONTEND_OP_AXIS_SYMBOL(op, type) (             \
            void* result, const void* a,                              \
            const int64_t *input_shape, size_t input_dims,            \
            const int64_t *output_shape, size_t output_dims,          \
            size_t axis)
#define ORT_MATH_FRONTEND_SERIAL_REDUCTION_AXIS_OP_DECL(op, type)     \
    ORT_MATH_KERNEL_ALIGN                                             \
        void ORT_MATH_FRONTEND_OP_AXIS_SYMBOL(serial_##op, type) (    \
            void* result, const void* a,                              \
            const int64_t *input_shape, size_t input_dims,            \
            const int64_t *output_shape, size_t output_dims,          \
            size_t axis)

#define ORT_MATH_FRONTEND_TRANSFORMATION_AXIS_OP_DECL(op, type)       \
    ORT_MATH_KERNEL_ALIGN                                             \
        void ORT_MATH_FRONTEND_OP_AXIS_SYMBOL(op, type) (             \
            void* result, const void* a,                              \
            const int64_t *shape, size_t dims,                        \
            size_t axis)

#define ORT_MATH_FRONTEND_BINARY_OP_IMPL(op_name, c_type, operator)   \
    ORT_MATH_FRONTEND_BINARY_OP_DECL(op_name, c_type) {               \
    c_type* res = (c_type*)result;                                    \
    const c_type* va = (const c_type*)a;                              \
    const c_type* vb = (const c_type*)b;                              \
    for (size_t i = 0; i < count; i++) {                              \
        res[i] = va[i] operator vb[i];                                \
    }                                                                 \
}

#define ORT_MATH_FRONTEND_UNARY_OP_IMPL(op_name, c_type, operator)    \
    ORT_MATH_FRONTEND_UNARY_OP_DECL(op_name, c_type) {                \
    c_type* res = (c_type*)result;                                    \
    const c_type* va = (const c_type*)a;                              \
    for (size_t i = 0; i < count; i++) {                              \
        res[i] = operator va[i];                                      \
    }                                                                 \
}

#define ORT_MATH_FRONTEND_SCALAR_OP_IMPL(op_name, c_type, operator)   \
    ORT_MATH_FRONTEND_BINARY_OP_DECL(op_name##_scalar, c_type) {      \
    c_type* res = (c_type*)result;                                    \
    const c_type* va = (const c_type*)a;                              \
    c_type sb = *(const c_type*)b;                                    \
    for (size_t i = 0; i < count; i++) {                              \
        res[i] = va[i] operator sb;                                   \
    }                                                                 \
}

#define ORT_MATH_FRONTEND_DISPATCH_SYMBOL(op_name)        \
    ort_math_frontend_dispatch_##op_name
#define ORT_MATH_FRONTEND_DISPATCH_DECL(op_kind, op_name) \
    static zend_always_inline op_kind                     \
        ORT_MATH_FRONTEND_DISPATCH_SYMBOL(op_name)(       \
            ort_math_promotion_t* promotion,              \
            const ort_math_promotion_schema_t* schema)
#define ORT_MATH_FRONTEND_DISPATCH_RESULT_TYPE_IMPL(op_kind, op_name) \
    ORT_MATH_FRONTEND_DISPATCH_DECL(op_kind, op_name) {               \
        const ort_math_dispatch_t* dispatch =                         \
            ort_math_dispatch_type(                                   \
                promotion->result_type);                              \
        return (op_kind) dispatch->op_name##_func;                    \
    }
#endif