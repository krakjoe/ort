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

#ifndef HAVE_ORT_MATHS_CAST
#define HAVE_ORT_MATHS_CAST

#include "maths/core.h"

/**
 * Cast a buffer from one type to another, will parallelize the operation
 * if the count is large enough.
 *
 * @param src The source buffer.
 * @param dst The destination buffer.
 * @param src_type The source type.
 * @param dst_type The destination type.
 * @param count The number of elements to cast.
 */
void ort_math_cast_buffer(
    const void* src, void* dst,
    ONNXTensorElementDataType src_type,
    ONNXTensorElementDataType dst_type, size_t count);

/**
 * Cast an element from one type to another.
 *
 * @param src The source element.
 * @param dst The destination element.
 * @param src_type The source type.
 * @param dst_type The destination type.
 */
void ort_math_cast_element(
    const void* src, void* dst, 
    ONNXTensorElementDataType src_type, 
    ONNXTensorElementDataType dst_type);

#endif