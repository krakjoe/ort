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

#ifdef _WIN32
#include <intrin.h>
#define __cpuid_count(leaf, level, eax, ebx, ecx, edx) do { \
    int registers[4];   \
    __cpuidex(          \
        registers,      \
        leaf, level);   \
    eax = registers[0]; \
    ebx = registers[1]; \
    ecx = registers[2]; \
    edx = registers[3]; \
} while(0)
#else
#include <cpuid.h>
#endif

typedef enum {
    ORT_MATH_BACKEND_NONE = 0,
    ORT_MATH_BACKEND_AVX2,
    ORT_MATH_BACKEND_SSE2,
    ORT_MATH_BACKEND_SSE41,
} ort_math_backend_type_t;

#define __ORT_MATH_BACKEND_CPU_TOPOLOGY      0x1A
#define __ORT_MATH_BACKEND_CPU_SHIFT         24
#define __ORT_MATH_BACKEND_CPU_MASK          0xFF
#define __ORT_MATH_BACKEND_CPU_GRACEMONT     0x20

#define __ORT_MATH_BACKEND_CPU_BIT_SSE2      (1 << 26)
#define __ORT_MATH_BACKEND_CPU_BIT_SSE41     (1 << 19)
#define __ORT_MATH_BACKEND_CPU_BIT_AVX       (1 << 28)
#define __ORT_MATH_BACKEND_CPU_BIT_AVX2      (1 << 5)
#define __ORT_MATH_BACKEND_CPU_BIT_OSXSAVE   (1 << 27)

#define __ORT_MATH_BACKEND_CPU_LEAF_BASIC    1
#define __ORT_MATH_BACKEND_CPU_LEAF_EXTENDED 7

#define __ORT_MATH_BACKEND_CPU_EXT           (0x6ULL)

#define __ORT_MATH_BACKEND_CPU_BITS(reg, bits) \
    ((reg) & (bits)) == (bits)
#define __ORT_MATH_BACKEND_CPU_FEATURES(eax, edx, bits) \
    ((((uint64_t)edx << 32) | eax) & (bits)) == (bits)

static zend_always_inline zend_bool __ort_math_backend_ecore() {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax = 0,
                 ebx = 0, 
                 ecx = 0, 
                 edx = 0;

    __cpuid_count(0, 0, eax, ebx, ecx, edx);
    if (eax < __ORT_MATH_BACKEND_CPU_TOPOLOGY)
        // No hybrid topology present, safe to
        // assume this isn't an ecore 
        return 0;

    // Read the CPU type
    __cpuid_count(
        __ORT_MATH_BACKEND_CPU_TOPOLOGY,
        0, eax, ebx, ecx, edx);
    unsigned int type =
        (eax >> __ORT_MATH_BACKEND_CPU_SHIFT) &
            __ORT_MATH_BACKEND_CPU_MASK;

    // Known E-core types (may need updating for future CPUs)
    return type == __ORT_MATH_BACKEND_CPU_GRACEMONT;
#else
    return 0;
#endif
}

/** Shall return true if the current thread should be 
 *      guarded from features provided by the given backend type */
static zend_always_inline zend_bool
    __ort_math_backend_guard(
        ort_math_backend_type_t type) {
    if (type == ORT_MATH_BACKEND_NONE) {
        return 1;
    }

    if (type == ORT_MATH_BACKEND_AVX2 && __ort_math_backend_ecore()) {
        return 1;
    }

    /* use cpuid_count to check feature support on this core for the given type */
    unsigned int eax = 0, 
                 ebx = 0,
                 ecx = 0, 
                 edx = 0;
    switch (type) {
        case ORT_MATH_BACKEND_AVX2: {
            // Check if CPUID supports extended features
            __cpuid_count(0, 0,
                eax, ebx, ecx, edx);
            if (eax < __ORT_MATH_BACKEND_CPU_LEAF_EXTENDED) {
                return 1;
            }

            __cpuid_count(
                __ORT_MATH_BACKEND_CPU_LEAF_BASIC,
                0, eax, ebx, ecx, edx);

            // Check for OSXSAVE
            if (!__ORT_MATH_BACKEND_CPU_BITS(ecx,
                    __ORT_MATH_BACKEND_CPU_BIT_OSXSAVE)) {
                return 1;
            }

            // Check for AVX
            if (!__ORT_MATH_BACKEND_CPU_BITS(ecx,
                    __ORT_MATH_BACKEND_CPU_BIT_AVX)) {
                return 1;
            }

            // Check XMM/YMM state saving support
#ifdef _WIN32
            uint64_t xcr0 = _xgetbv(0);

            if (!__ORT_MATH_BACKEND_CPU_BITS(xcr0,
                    __ORT_MATH_BACKEND_CPU_EXT)) {
                return 1;
            }
#else
            __asm__ volatile (
                "xgetbv"
                : "=a"(eax), "=d"(edx)
                : "c" (0)
            );

            if (!__ORT_MATH_BACKEND_CPU_FEATURES(
                    eax, edx,
                    __ORT_MATH_BACKEND_CPU_EXT)) {
                return 1;
            }
#endif
            // Check AVX2 support
            __cpuid_count(
                __ORT_MATH_BACKEND_CPU_LEAF_EXTENDED, 
                0, eax, ebx, ecx, edx);

            return !__ORT_MATH_BACKEND_CPU_BITS(ebx,
                __ORT_MATH_BACKEND_CPU_BIT_AVX2);
        }
        case ORT_MATH_BACKEND_SSE41: {
            __cpuid_count(
                __ORT_MATH_BACKEND_CPU_LEAF_BASIC, 
                0, eax, ebx, ecx, edx);

            return !__ORT_MATH_BACKEND_CPU_BITS(ecx,
                __ORT_MATH_BACKEND_CPU_BIT_SSE41);
        }
        case ORT_MATH_BACKEND_SSE2: {
            __cpuid_count(
                __ORT_MATH_BACKEND_CPU_LEAF_BASIC,
                0, eax, ebx, ecx, edx);

            return !__ORT_MATH_BACKEND_CPU_BITS(edx,
                __ORT_MATH_BACKEND_CPU_BIT_SSE2);
        }
    }

    /* safety: unknown backends should be guarded */
    return 1;
}

#undef __ORT_MATH_BACKEND_CPU_TOPOLOGY
#undef __ORT_MATH_BACKEND_CPU_SHIFT
#undef __ORT_MATH_BACKEND_CPU_MASK
#undef __ORT_MATH_BACKEND_CPU_GRACEMONT

#undef __ORT_MATH_BACKEND_CPU_BIT_SSE2
#undef __ORT_MATH_BACKEND_CPU_BIT_SSE41
#undef __ORT_MATH_BACKEND_CPU_BIT_AVX
#undef __ORT_MATH_BACKEND_CPU_BIT_AVX2
#undef __ORT_MATH_BACKEND_CPU_BIT_OSXSAVE

#undef __ORT_MATH_BACKEND_CPU_LEAF_BASIC
#undef __ORT_MATH_BACKEND_CPU_LEAF_EXTENDED

#undef __ORT_MATH_BACKEND_CPU_EXT
#undef __ORT_MATH_BACKEND_CPU_BITS
#undef __ORT_MATH_BACKEND_CPU_FEATURES

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