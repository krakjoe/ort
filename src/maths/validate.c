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

#include "status.h"

#include "maths/validate.h"

zend_bool ort_math_validate_input(ort_tensor_t* tensor, const char* operation_name) {
    php_ort_status_flow(
        (!tensor || !tensor->data),
        return 0,
        php_ort_status_tensor_invaliddata_ce,
        "%s: tensor has no data", operation_name);

    php_ort_status_flow(
        (tensor->elements == 0),
        return 0,
        php_ort_status_tensor_invalidshape_ce,
        "%s: tensor is empty", operation_name);

    php_ort_status_flow(
        (tensor->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64),
        return 0,
        php_ort_status_tensor_invalidtype_ce,
        "%s: UINT64 tensor type is not supported", operation_name);
    
    return 1;
}

zend_bool ort_math_validate_axis(ort_tensor_t* tensor, zend_long axis, const char* operation_name) {
    if (axis < 0) {
        axis += tensor->dimensions;
    }

    php_ort_status_flow(
        (axis < 0 || axis >= tensor->dimensions),
        return 0,
        php_ort_status_tensor_invalidshape_ce,
        "%s: axis %ld is out of bounds for tensor with %zu dimensions", 
        operation_name, axis, tensor->dimensions);

    return 1;
}
