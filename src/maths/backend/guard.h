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

#ifndef HAVE_ORT_MATHS_BACKEND_GUARD
#define HAVE_ORT_MATHS_BACKEND_GUARD

/* {{{
    On platforms with a hybrid layout of P and E cores, it's necessary to guard
    against executing code that is not supported on the current core type.

    On platforms where the CPU is homogeneous (for example neon on ARM), simd instructions
    may still be disabled by kernel parameters (nosimd).

    In every case, these guards are invoked once for each thread where the backend may operate.
    Whatever the cost it's worth it to ensure stability in server environments, and to ensure that
    where someone is engaged in a debugging workflow (ie, using nosimd or something like it), ort
    isn't a red herring.

    The installation routine is called once for each thread where the backend may operate.
        - The main thread
        - Pool threads

    In every case, where backend is called the thread will be pinned to that core, to
    provide the guarantee that the thread will not migrate to a different core, and the fast
    paths it installs for that thread (in the allocator and dispatch table) remain coherent.

    I'm ever so sorry about this code; I've tried to make it as readable/logical and maintanable
    as possible; however, what is going on here is awkward and varies across platform.

    I haven't documented every mysterious thing going on, but have named the magic numbers which
    you should be able to research if you want to find out what it means.
*/

#include <stdbool.h>
#include <stdint.h>

#if defined(__aarch64__)
#if !defined(__APPLE__) && !defined(__MACH__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif
/* No additional includes needed for macOS ARM64 */
#elif defined(_WIN32)
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
    ORT_MATH_BACKEND_AVX512,
    ORT_MATH_BACKEND_SSE2,
    ORT_MATH_BACKEND_SSE41,
    ORT_MATH_BACKEND_NEON,
    ORT_MATH_BACKEND_RISCV64,
    ORT_MATH_BACKEND_WASM,
} ort_math_backend_type_t;

#define ORT_MATH_BACKEND_TYPE(TYPE) ORT_MATH_BACKEND_##TYPE

#ifdef __aarch64__
#define __ORT_MATH_BACKEND_CPU_FEATURES      AT_HWCAP
#define __ORT_MATH_BACKEND_CPU_NEON          HWCAP_ASIMD
#else
#define __ORT_MATH_BACKEND_CPU_TOPOLOGY      0x1A
#define __ORT_MATH_BACKEND_CPU_SHIFT         24
#define __ORT_MATH_BACKEND_CPU_MASK          0xFF
#define __ORT_MATH_BACKEND_CPU_GRACEMONT     0x20

#define __ORT_MATH_BACKEND_CPU_BIT_SSE2      (1 << 26)
#define __ORT_MATH_BACKEND_CPU_BIT_SSE41     (1 << 19)
#define __ORT_MATH_BACKEND_CPU_BIT_AVX       (1 << 28)
#define __ORT_MATH_BACKEND_CPU_BIT_AVX2      (1 << 5)
#define __ORT_MATH_BACKEND_CPU_BIT_AVX512    (1 << 16)
#define __ORT_MATH_BACKEND_CPU_BIT_OSXSAVE   (1 << 27)

#define __ORT_MATH_BACKEND_CPU_LEAF_BASIC    1
#define __ORT_MATH_BACKEND_CPU_LEAF_EXTENDED 7

#define __ORT_MATH_BACKEND_CPU_EXT           (0x6ULL)
#define __ORT_MATH_BACKEND_CPU_FEATURES(eax, edx, bits) \
    (((((uint64_t)edx << 32) | eax) & (bits)) == (bits))
#endif

#define __ORT_MATH_BACKEND_CPU_BITS(reg, bits) \
   (((reg) & (bits)) == (bits))

static inline bool __ort_math_backend_ecore(void) {
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
static inline bool
    __ort_math_backend_guard(
        ort_math_backend_type_t type) {
    if (type == ORT_MATH_BACKEND_NONE) {
        return 1;
    }

    if (type == ORT_MATH_BACKEND_RISCV64 ||
        type == ORT_MATH_BACKEND_WASM) {
        return 0;
    }

#if defined(__aarch64__)
#if defined(__APPLE__) || defined(__MACH__)
    /* On Apple Silicon (macOS ARM64), NEON is always available */
    switch(type) {
        case ORT_MATH_BACKEND_NEON:
            return 0; /* NEON is always available, so never guard */
        default:
            return 1;
    }
#else
    unsigned long features = getauxval(
        __ORT_MATH_BACKEND_CPU_FEATURES);
    switch(type) {
        case ORT_MATH_BACKEND_NEON:
            return !__ORT_MATH_BACKEND_CPU_BITS(
                features, __ORT_MATH_BACKEND_CPU_NEON);
        default:
            return 1;
    }
#endif
#else
    if ((type == ORT_MATH_BACKEND_AVX2 ||
         type == ORT_MATH_BACKEND_AVX512) && __ort_math_backend_ecore()) {
        return 1;
    }

    /* use cpuid_count to check feature support on this core for the given type */
    unsigned int eax = 0, 
                 ebx = 0,
                 ecx = 0, 
                 edx = 0;
    switch (type) {
        case ORT_MATH_BACKEND_AVX512:
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

            if (!__ORT_MATH_BACKEND_CPU_BITS(ebx,
                 __ORT_MATH_BACKEND_CPU_BIT_AVX2)) {
                return 1;
            }

            if (type == ORT_MATH_BACKEND_AVX512) {
                // Check AVX512F support
                return !__ORT_MATH_BACKEND_CPU_BITS(ebx,
                    __ORT_MATH_BACKEND_CPU_BIT_AVX512);
            }

            // AVX2 support confirmed
            return 0;
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

        default: {}
    }
#endif
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
#undef __ORT_MATH_BACKEND_CPU_BIT_AVX512
#undef __ORT_MATH_BACKEND_CPU_BIT_OSXSAVE

#undef __ORT_MATH_BACKEND_CPU_LEAF_BASIC
#undef __ORT_MATH_BACKEND_CPU_LEAF_EXTENDED

#undef __ORT_MATH_BACKEND_CPU_EXT
#undef __ORT_MATH_BACKEND_CPU_BITS
#undef __ORT_MATH_BACKEND_CPU_FEATURES

#define ORT_MATH_BACKEND_GUARD __ort_math_backend_guard /* }}} */
#endif
