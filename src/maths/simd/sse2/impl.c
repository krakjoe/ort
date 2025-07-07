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

#include "maths/simd/impl.h"

void ort_math_simd_install(ort_math_type_dispatch_t* table) {
    /* abs.c */
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)].abs_func     = ort_math_simd_abs_float;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)].abs_func    = ort_math_simd_abs_double;

    /* add.c */
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)].add_func     = ort_math_simd_add_float;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)].add_func    = ort_math_simd_add_double;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)].add_func      = ort_math_simd_add_int8_t;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)].add_func     = ort_math_simd_add_int16_t;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)].add_func     = ort_math_simd_add_int32_t;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)].add_func     = ort_math_simd_add_uint8_t;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)].add_func    = ort_math_simd_add_uint16_t;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)].add_func    = ort_math_simd_add_uint32_t;

    /* div.c */
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)].div_func     = ort_math_simd_div_float;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)].div_func    = ort_math_simd_div_double;

    /* matmul.c */
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)].matmul_func  = ort_math_simd_matmul_float;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)].matmul_func = ort_math_simd_matmul_double;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)].matmul_func  = ort_math_simd_matmul_int16_t;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)].mul_func    = ort_math_simd_mul_uint16_t;

    /* neg.c */
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)].neg_func     = ort_math_simd_neg_float;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)].neg_func    = ort_math_simd_neg_double;

    /* recip.c */
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)].recip_func   = ort_math_simd_recip_float;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)].recip_func  = ort_math_simd_recip_double;

    /* sign.c */
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)].sign_func    = ort_math_simd_sign_float;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)].sign_func   = ort_math_simd_sign_double;

    /* sqrt.c */
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)].sqrt_func    = ort_math_simd_sqrt_float;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)].sqrt_func   = ort_math_simd_sqrt_double;

    /* sub.c */
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)].sub_func     = ort_math_simd_sub_float;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)].sub_func    = ort_math_simd_sub_double;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)].sub_func      = ort_math_simd_sub_int8_t;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)].sub_func     = ort_math_simd_sub_int16_t;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)].sub_func     = ort_math_simd_sub_int32_t;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)].sub_func     = ort_math_simd_sub_int64_t;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)].sub_func     = ort_math_simd_sub_uint8_t;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)].sub_func    = ort_math_simd_sub_uint16_t;
    table[ort_math_type_dispatch_indexof(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)].sub_func    = ort_math_simd_sub_uint32_t;
}
