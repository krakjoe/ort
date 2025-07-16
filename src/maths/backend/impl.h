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

/* {{{
    On platforms with a hybrid layout of P and E cores, it's necessary to guard
    against executing code that is not supported on the current core type.

    The installation routine is called once for each thread where the backend may operate.
        - The main thread
        - Pool threads

    In every case, where backend is called the thread will be pinned to that core, to
    provide the guarantee that the thread will not migrate to a different core, and the fast
    paths it installs for that thread (in the allocator and dispatch table) remain coherent.
*/
typedef enum {
    ORT_MATH_BACKEND_NONE = 0,
    ORT_MATH_BACKEND_AVX2,
    ORT_MATH_BACKEND_SSE2,
    ORT_MATH_BACKEND_SSE41,
} ort_math_backend_type_t;

/** {{{ Shall return true if the current core should be guarded from the given backend type }}} */
static zend_always_inline zend_bool
    __ort_math_backend_guard(
        ort_math_backend_type_t type) {
    if (ORT_MATH_BACKEND_NONE == type) {
        return 1;
    }

#ifndef _WIN32
    switch (type) {
        case ORT_MATH_BACKEND_AVX2:
            return (zend_bool)!__builtin_cpu_supports("avx2");
        case ORT_MATH_BACKEND_SSE2:
            return (zend_bool)!__builtin_cpu_supports("sse2");
        case ORT_MATH_BACKEND_SSE41:
            return (zend_bool)!__builtin_cpu_supports("sse4.1");
    }
#else
    int cpuInfo[4] = {0};
    switch (type) {
        case ORT_MATH_BACKEND_AVX2:
            /* Check AVX2 support: CPUID leaf 7, EBX bit 5 */
            __cpuid(cpuInfo, 0);

            if (cpuInfo[0] < 7)
                return 1;

            __cpuid(cpuInfo, 1);

            int osxsave =
                (cpuInfo[2] & (1 << 27)) != 0;
            int avx =
                (cpuInfo[2] & (1 << 28)) != 0;
            if (!osxsave || !avx)
                return 1; /* OSXSAVE and AVX must be supported */
            unsigned long long xcrFeatureMask = 0;
            
            xcrFeatureMask = _xgetbv(0);
            if ((xcrFeatureMask & 0x6) != 0x6)
                return 1;
            
            __cpuid(cpuInfo, 7);

            if ((cpuInfo[1] & (1 << 5)) == 0)
                return 1;

            return 0;
        case ORT_MATH_BACKEND_SSE2:
            __cpuid(cpuInfo, 1);
            
            if ((cpuInfo[3] & (1 << 26)) == 0)
                return 1; /* SSE2: EDX bit 26 */
            return 0;
        case ORT_MATH_BACKEND_SSE41:
            __cpuid(cpuInfo, 1);

            if ((cpuInfo[2] & (1 << 19)) == 0)
                return 1; /* SSE4.1: ECX bit 19 */
            return 0;
    }
#endif

    return 1;
}

#define ORT_MATH_BACKEND_GUARD __ort_math_backend_guard /* }}} */

/* {{{ 
    Each backend must call this macro within their install routine 
    to setup the dispatch table.
    This must never be invoked outside of the backend install function. */
#define ORT_MATH_BACKEND_INSTALL(table, onnx_type, func, c_type) \
    table[ort_math_dispatch_indexof(                             \
        ONNX_TENSOR_ELEMENT_DATA_TYPE_##onnx_type                \
    )].func##_func = ORT_MATH_BACKEND_OP_SYMBOL(func, c_type);   /* }}} */
#endif