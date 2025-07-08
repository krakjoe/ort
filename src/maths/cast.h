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

void ort_math_cast_element(
    const void* src, void* dst, 
    ONNXTensorElementDataType src_type, 
    ONNXTensorElementDataType dst_type);

/* Cast a buffer of elements from src_type to dst_type. Returns 1 on success, 0 on failure. */
static zend_always_inline zend_bool ort_math_cast_buffer(
    const void* src, void* dst,
    ONNXTensorElementDataType src_type,
    ONNXTensorElementDataType dst_type, size_t count) {
    if (src_type == dst_type) {
        size_t size = count * 
            php_ort_type_sizeof(dst_type);
        memcpy(dst, src, size);
        return 1;
    }

    for (size_t i = 0; i < count; i++) {
        const void* src_elem = 
            (const char*) src + i * php_ort_type_sizeof(src_type);
        void* dst_elem = 
            (char*)dst + i * php_ort_type_sizeof(dst_type);

        ort_math_cast_element(src_elem, dst_elem, src_type, dst_type);
    }

    return 1;
}

#endif