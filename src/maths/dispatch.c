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

#include "maths/dispatch.h"
#include "maths/frontend.h"

/* Complete dispatch table with SIMD optimization */
static ort_math_dispatch_t __ort_math_dispatch_table[] = {
    /* FLOAT */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        .add_func        = ort_math_ops_add_float,
        .sub_func        = ort_math_ops_sub_float,
        .mul_func        = ort_math_ops_mul_float,
        .div_func        = ort_math_ops_div_float,
        .mod_func        = ort_math_ops_mod_float,
        .pow_func        = ort_math_ops_pow_float,
        .ceil_func       = ort_math_ops_ceil_float,
        .floor_func      = ort_math_ops_floor_float,
        .round_func      = ort_math_ops_round_float,
        .abs_func        = ort_math_ops_abs_float,
        .sqrt_func       = ort_math_ops_sqrt_float,
        .neg_func        = ort_math_ops_neg_float,
        .recip_func      = ort_math_ops_recip_float,
        .sign_func       = ort_math_ops_sign_float,
        .trunc_func      = ort_math_ops_trunc_float,
        .dot_func        = ort_math_ops_dot_float,
        .matmul_func     = ort_math_ops_matmul_float,
    },
    /* DOUBLE */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        .add_func        = ort_math_ops_add_double,
        .sub_func        = ort_math_ops_sub_double,
        .mul_func        = ort_math_ops_mul_double,
        .div_func        = ort_math_ops_div_double,
        .mod_func        = ort_math_ops_mod_double,
        .pow_func        = ort_math_ops_pow_double,
        .ceil_func       = ort_math_ops_ceil_double,
        .floor_func      = ort_math_ops_floor_double,
        .round_func      = ort_math_ops_round_double,
        .abs_func        = ort_math_ops_abs_double,
        .sqrt_func       = ort_math_ops_sqrt_double,
        .neg_func        = ort_math_ops_neg_double,
        .recip_func      = ort_math_ops_recip_double,
        .sign_func       = ort_math_ops_sign_double,
        .trunc_func      = ort_math_ops_trunc_double,
        .dot_func        = ort_math_ops_dot_double,
        .matmul_func     = ort_math_ops_matmul_double,
    },
    /* INT8 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        .add_func        = ort_math_ops_add_int8_t,
        .sub_func        = ort_math_ops_sub_int8_t,
        .mul_func        = ort_math_ops_mul_int8_t,
        .div_func        = ort_math_ops_div_int8_t,
        .mod_func        = ort_math_ops_mod_int8_t,
        .pow_func        = ort_math_ops_pow_int8_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_int8_t,
        .neg_func        = ort_math_ops_neg_int8_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_int8_t,
        .matmul_func     = ort_math_ops_matmul_int8_t,
    },
    /* INT16 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
        .add_func        = ort_math_ops_add_int16_t,
        .sub_func        = ort_math_ops_sub_int16_t,
        .mul_func        = ort_math_ops_mul_int16_t,
        .div_func        = ort_math_ops_div_int16_t,
        .mod_func        = ort_math_ops_mod_int16_t,
        .pow_func        = ort_math_ops_pow_int16_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_int16_t,
        .neg_func        = ort_math_ops_neg_int16_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_int16_t,
        .matmul_func     = ort_math_ops_matmul_int16_t,
    },
    /* INT32 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        .add_func        = ort_math_ops_add_int32_t,
        .sub_func        = ort_math_ops_sub_int32_t,
        .mul_func        = ort_math_ops_mul_int32_t,
        .div_func        = ort_math_ops_div_int32_t,
        .mod_func        = ort_math_ops_mod_int32_t,
        .pow_func        = ort_math_ops_pow_int32_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_int32_t,
        .neg_func        = ort_math_ops_neg_int32_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_int32_t,
        .matmul_func     = ort_math_ops_matmul_int32_t,
    },
    /* INT64 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        .add_func        = ort_math_ops_add_int64_t,
        .sub_func        = ort_math_ops_sub_int64_t,
        .mul_func        = ort_math_ops_mul_int64_t,
        .div_func        = ort_math_ops_div_int64_t,
        .mod_func        = ort_math_ops_mod_int64_t,
        .pow_func        = ort_math_ops_pow_int64_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_int64_t,
        .neg_func        = ort_math_ops_neg_int64_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_int64_t,
        .matmul_func     = ort_math_ops_matmul_int64_t,
    },
    /* UINT8 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        .add_func        = ort_math_ops_add_uint8_t,
        .sub_func        = ort_math_ops_sub_uint8_t,
        .mul_func        = ort_math_ops_mul_uint8_t,
        .div_func        = ort_math_ops_div_uint8_t,
        .mod_func        = ort_math_ops_mod_uint8_t,
        .pow_func        = ort_math_ops_pow_uint8_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_uint8_t,
        .neg_func        = ort_math_ops_neg_uint8_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_uint8_t,
        .matmul_func     = ort_math_ops_matmul_uint8_t,
    },
    /* UINT16 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
        .add_func        = ort_math_ops_add_uint16_t,
        .sub_func        = ort_math_ops_sub_uint16_t,
        .mul_func        = ort_math_ops_mul_uint16_t,
        .div_func        = ort_math_ops_div_uint16_t,
        .mod_func        = ort_math_ops_mod_uint16_t,
        .pow_func        = ort_math_ops_pow_uint16_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_uint16_t,
        .neg_func        = ort_math_ops_neg_uint16_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_uint16_t,
        .matmul_func     = ort_math_ops_matmul_uint16_t,
    },
    /* UINT32 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
        .add_func        = ort_math_ops_add_uint32_t,
        .sub_func        = ort_math_ops_sub_uint32_t,
        .mul_func        = ort_math_ops_mul_uint32_t,
        .div_func        = ort_math_ops_div_uint32_t,
        .mod_func        = ort_math_ops_mod_uint32_t,
        .pow_func        = ort_math_ops_pow_uint32_t,
        .ceil_func       = NULL,
        .floor_func      = NULL,
        .round_func      = NULL,
        .abs_func        = NULL,
        .sqrt_func       = ort_math_ops_sqrt_uint32_t,
        .neg_func        = ort_math_ops_neg_uint32_t,
        .recip_func      = NULL,
        .sign_func       = NULL,
        .trunc_func      = NULL,
        .dot_func        = ort_math_ops_dot_uint32_t,
        .matmul_func     = ort_math_ops_matmul_uint32_t,
    },
    /* BOOL */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
        .add_func        = ort_math_ops_add_zend_bool,   // Logical OR (or addition)
        .sub_func        = ort_math_ops_sub_zend_bool,   // Logical XOR (or subtraction)
        .mul_func        = ort_math_ops_mul_zend_bool,   // Logical AND (or multiplication)
        .div_func        = ort_math_ops_div_zend_bool,   // Logical division (A && B)
        .mod_func        = NULL,                         // Logical modulo (A && !B)
        .pow_func        = NULL,                         // Logical power (A ** B)
        .ceil_func       = NULL,                         // Not meaningful for bool
        .floor_func      = NULL,                         // Not meaningful for bool
        .round_func      = NULL,                         // Not meaningful for bool
        .abs_func        = NULL,                         // Identity for bool
        .sqrt_func       = NULL,                         // Not meaningful for bool
        .neg_func        = ort_math_ops_neg_zend_bool,   // Logical NOT
        .recip_func      = NULL,                         // Not meaningful for bool
        .sign_func       = NULL,                         // Identity for bool
        .trunc_func      = NULL,                         // Not meaningful for bool
        .dot_func        = NULL,
        .matmul_func     = NULL,
    }
};

ort_math_dispatch_t* ort_math_dispatch_table(void) {
    return __ort_math_dispatch_table;
}

const ort_math_dispatch_t* ort_math_dispatch_type(
    ONNXTensorElementDataType type) {
    return &__ort_math_dispatch_table[
        ort_math_dispatch_indexof(type)
    ];
}
