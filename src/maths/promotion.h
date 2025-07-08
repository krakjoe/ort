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

#ifndef HAVE_ORT_MATHS_PROMOTION
#define HAVE_ORT_MATHS_PROMOTION

#include "maths/core.h"

typedef struct _ort_math_type_promotion_t {
    ONNXTensorElementDataType result_type;
    zend_bool is_valid;
} ort_math_type_promotion_t;

ort_math_type_promotion_t ort_math_type_promote(
    ONNXTensorElementDataType type_a,
    ONNXTensorElementDataType type_b
);

ort_math_type_promotion_t ort_math_type_promote_strict(
    ONNXTensorElementDataType type_a,
    ONNXTensorElementDataType type_b
);

#endif