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

#include <stdint.h>
#include "maths/float16.h"

extern int __ort_cuda_threads;

static inline int ort_cuda_blocks_count(const int threads, size_t count) {
    return (count + threads - 1) / threads;
}

static inline int ort_cuda_blocks_min(const int threads, size_t count, const int minimum) {
    return (int) min(ort_cuda_blocks_count(threads, count), minimum);
}

#define ort_cuda_half_from_float16(arg) __short_as_half((short) arg)
#define ort_cuda_float16_from_half(arg) (float16) __half_as_short(arg)