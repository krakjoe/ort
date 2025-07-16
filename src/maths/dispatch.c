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
#include "maths/frontend/impl.h"

/* Error dispatch */
ORT_TLS ort_math_dispatch_t __ort_math_dispatch_error;
 
/* Complete dispatch table */
ORT_TLS ort_math_dispatch_t __ort_math_dispatch_table[] = {
    /* FLOAT */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        .add_func        = ort_math_frontend_add_float,
        .add_scalar_func = ort_math_frontend_add_scalar_float,

        .div_func        = ort_math_frontend_div_float,
        .div_scalar_func = ort_math_frontend_div_scalar_float,

        .sub_func        = ort_math_frontend_sub_float,
        .sub_scalar_func = ort_math_frontend_sub_scalar_float,

        .mul_func        = ort_math_frontend_mul_float,
        .mul_scalar_func = ort_math_frontend_mul_scalar_float,

        .ceil_func       = ort_math_frontend_ceil_float,
        .floor_func      = ort_math_frontend_floor_float,
        .round_func      = ort_math_frontend_round_float,
        .abs_func        = ort_math_frontend_abs_float,
        .sqrt_func       = ort_math_frontend_sqrt_float,
        .neg_func        = ort_math_frontend_neg_float,
        .recip_func      = ort_math_frontend_recip_float,
        .sign_func       = ort_math_frontend_sign_float,
        .trunc_func      = ort_math_frontend_trunc_float,
        .dot_func        = ort_math_frontend_dot_float,
        .matmul_func     = ort_math_frontend_matmul_float,
        .sum_func        = ort_math_frontend_sum_float,
        .sum_axis_func   = ort_math_frontend_sum_axis_float,
        .min_func        = ort_math_frontend_min_float,
        .min_axis_func   = ort_math_frontend_min_axis_float,
        .max_func        = ort_math_frontend_max_float,
        .max_axis_func   = ort_math_frontend_max_axis_float,
        .mean_func       = ort_math_frontend_mean_float,
        .mean_axis_func  = ort_math_frontend_mean_axis_float,
        .softmax_axis_func = ort_math_frontend_softmax_axis_float,

        .mod_func        = ort_math_frontend_mod_float,
        .mod_scalar_func = ort_math_frontend_mod_scalar_float,

        .pow_func        = ort_math_frontend_pow_float,
        .pow_scalar_func = ort_math_frontend_pow_scalar_float,
    },
    /* DOUBLE */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        .add_func        = ort_math_frontend_add_double,
        .add_scalar_func = ort_math_frontend_add_scalar_double,

        .div_func        = ort_math_frontend_div_double,
        .div_scalar_func = ort_math_frontend_div_scalar_double,

        .sub_func        = ort_math_frontend_sub_double,
        .sub_scalar_func = ort_math_frontend_sub_scalar_double,

        .mul_func        = ort_math_frontend_mul_double,
        .mul_scalar_func = ort_math_frontend_mul_scalar_double,

        .ceil_func       = ort_math_frontend_ceil_double,
        .floor_func      = ort_math_frontend_floor_double,
        .round_func      = ort_math_frontend_round_double,
        .abs_func        = ort_math_frontend_abs_double,
        .sqrt_func       = ort_math_frontend_sqrt_double,
        .neg_func        = ort_math_frontend_neg_double,
        .recip_func      = ort_math_frontend_recip_double,
        .sign_func       = ort_math_frontend_sign_double,
        .trunc_func      = ort_math_frontend_trunc_double,
        .dot_func        = ort_math_frontend_dot_double,
        .matmul_func     = ort_math_frontend_matmul_double,
        .sum_func        = ort_math_frontend_sum_double,
        .sum_axis_func   = ort_math_frontend_sum_axis_double,
        .min_func        = ort_math_frontend_min_double,
        .min_axis_func   = ort_math_frontend_min_axis_double,
        .max_func        = ort_math_frontend_max_double,
        .max_axis_func   = ort_math_frontend_max_axis_double,
        .mean_func       = ort_math_frontend_mean_double,
        .mean_axis_func  = ort_math_frontend_mean_axis_double,
        .softmax_axis_func = ort_math_frontend_softmax_axis_double,

        .mod_func        = ort_math_frontend_mod_double,
        .mod_scalar_func = ort_math_frontend_mod_scalar_double,

        .pow_func        = ort_math_frontend_pow_double,
        .pow_scalar_func = ort_math_frontend_pow_scalar_double,
    },
    /* INT8 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
        .add_func        = ort_math_frontend_add_int8_t,
        .add_scalar_func = ort_math_frontend_add_scalar_int8_t,

        .div_func        = NULL,
        .div_scalar_func = NULL,

        .sub_func        = ort_math_frontend_sub_int8_t,
        .sub_scalar_func = ort_math_frontend_sub_scalar_int8_t,

        .mul_func        = ort_math_frontend_mul_int8_t,
        .mul_scalar_func = ort_math_frontend_mul_scalar_int8_t,

        .ceil_func       = ort_math_frontend_ceil_int8_t,
        .floor_func      = ort_math_frontend_floor_int8_t,
        .round_func      = ort_math_frontend_round_int8_t,
        .abs_func        = ort_math_frontend_abs_int8_t,
        .sqrt_func       = NULL,
        .neg_func        = ort_math_frontend_neg_int8_t,
        .recip_func      = NULL,
        .sign_func       = ort_math_frontend_sign_int8_t,
        .trunc_func      = ort_math_frontend_trunc_int8_t,
        .dot_func        = ort_math_frontend_dot_int8_t,
        .matmul_func     = ort_math_frontend_matmul_int8_t,
        .sum_func        = NULL,
        .sum_axis_func   = NULL,
        .min_func        = ort_math_frontend_min_int8_t,
        .min_axis_func   = ort_math_frontend_min_axis_int8_t,
        .max_func        = ort_math_frontend_max_int8_t,
        .max_axis_func   = ort_math_frontend_max_axis_int8_t,
        .mean_func       = NULL,
        .mean_axis_func  = NULL,
        .softmax_axis_func = NULL,

        .mod_func        = ort_math_frontend_mod_int8_t,
        .mod_scalar_func = ort_math_frontend_mod_scalar_int8_t,

        .pow_func        = ort_math_frontend_pow_int8_t,
        .pow_scalar_func = ort_math_frontend_pow_scalar_int8_t,
    },
    /* INT16 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
        .add_func        = ort_math_frontend_add_int16_t,
        .add_scalar_func = ort_math_frontend_add_scalar_int16_t,

        .div_func        = NULL,
        .div_scalar_func = NULL,

        .sub_func        = ort_math_frontend_sub_int16_t,
        .sub_scalar_func = ort_math_frontend_sub_scalar_int16_t,
    
        .mul_func        = ort_math_frontend_mul_int16_t,
        .mul_scalar_func = ort_math_frontend_mul_scalar_int16_t,

        .ceil_func       = ort_math_frontend_ceil_int16_t,
        .floor_func      = ort_math_frontend_floor_int16_t,
        .round_func      = ort_math_frontend_round_int16_t,
        .abs_func        = ort_math_frontend_abs_int16_t,
        .sqrt_func       = NULL,
        .neg_func        = ort_math_frontend_neg_int16_t,
        .recip_func      = NULL,
        .sign_func       = ort_math_frontend_sign_int16_t,
        .trunc_func      = ort_math_frontend_trunc_int16_t,
        .dot_func        = ort_math_frontend_dot_int16_t,
        .matmul_func     = ort_math_frontend_matmul_int16_t,
        .sum_func        = NULL,
        .sum_axis_func   = NULL,
        .min_func        = ort_math_frontend_min_int16_t,
        .min_axis_func   = ort_math_frontend_min_axis_int16_t,
        .max_func        = ort_math_frontend_max_int16_t,
        .max_axis_func   = ort_math_frontend_max_axis_int16_t,
        .mean_func       = NULL,
        .mean_axis_func  = NULL,
        .softmax_axis_func = NULL,

        .mod_func        = ort_math_frontend_mod_int16_t,
        .mod_scalar_func = ort_math_frontend_mod_scalar_int16_t,

        .pow_func        = ort_math_frontend_pow_int16_t,
        .pow_scalar_func = ort_math_frontend_pow_scalar_int16_t,
    },
    /* INT32 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
        .add_func        = ort_math_frontend_add_int32_t,
        .add_scalar_func = ort_math_frontend_add_scalar_int32_t,

        .div_func        = NULL,
        .div_scalar_func = NULL,

        .sub_func        = ort_math_frontend_sub_int32_t,
        .sub_scalar_func = ort_math_frontend_sub_scalar_int32_t,

        .mul_func        = ort_math_frontend_mul_int32_t,
        .mul_scalar_func = ort_math_frontend_mul_scalar_int32_t,

        .ceil_func       = ort_math_frontend_ceil_int32_t,
        .floor_func      = ort_math_frontend_floor_int32_t,
        .round_func      = ort_math_frontend_round_int32_t,
        .abs_func        = ort_math_frontend_abs_int32_t,
        .sqrt_func       = NULL,
        .neg_func        = ort_math_frontend_neg_int32_t,
        .recip_func      = NULL,
        .sign_func       = ort_math_frontend_sign_int32_t,
        .trunc_func      = ort_math_frontend_trunc_int32_t,
        .dot_func        = ort_math_frontend_dot_int32_t,
        .matmul_func     = ort_math_frontend_matmul_int32_t,
        .sum_func        = NULL,
        .sum_axis_func   = NULL,
        .min_func        = ort_math_frontend_min_int32_t,
        .min_axis_func   = ort_math_frontend_min_axis_int32_t,
        .max_func        = ort_math_frontend_max_int32_t,
        .max_axis_func   = ort_math_frontend_max_axis_int32_t,
        .mean_func       = NULL,
        .mean_axis_func  = NULL,
        .softmax_axis_func = NULL,
        
        .mod_func        = ort_math_frontend_mod_int32_t,
        .mod_scalar_func = ort_math_frontend_mod_scalar_int32_t,

        .pow_func        = ort_math_frontend_pow_int32_t,
        .pow_scalar_func = ort_math_frontend_pow_scalar_int32_t,
    },
    /* INT64 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        .add_func        = ort_math_frontend_add_int64_t,
        .add_scalar_func = ort_math_frontend_add_scalar_int64_t,

        .sub_func        = ort_math_frontend_sub_int64_t,
        .sub_scalar_func = ort_math_frontend_sub_scalar_int64_t,

        .mul_func        = ort_math_frontend_mul_int64_t,
        .mul_scalar_func = ort_math_frontend_mul_scalar_int64_t,
    
        .div_func        = NULL,
        .div_scalar_func = NULL,
        .ceil_func       = ort_math_frontend_ceil_int64_t,
        .floor_func      = ort_math_frontend_floor_int64_t,
        .round_func      = ort_math_frontend_round_int64_t,
        .abs_func        = ort_math_frontend_abs_int64_t,
        .sqrt_func       = NULL,
        .neg_func        = ort_math_frontend_neg_int64_t,
        .recip_func      = NULL,
        .sign_func       = ort_math_frontend_sign_int64_t,
        .trunc_func      = ort_math_frontend_trunc_int64_t,
        .dot_func        = ort_math_frontend_dot_int64_t,
        .matmul_func     = ort_math_frontend_matmul_int64_t,
        .sum_func        = ort_math_frontend_sum_int64_t,
        .sum_axis_func   = ort_math_frontend_sum_axis_int64_t,
        .min_func        = ort_math_frontend_min_int64_t,
        .min_axis_func   = ort_math_frontend_min_axis_int64_t,
        .max_func        = ort_math_frontend_max_int64_t,
        .max_axis_func   = ort_math_frontend_max_axis_int64_t,
        .mean_func       = NULL,
        .mean_axis_func  = NULL,
        .softmax_axis_func = NULL,

        .mod_func        = ort_math_frontend_mod_int64_t,
        .mod_scalar_func = ort_math_frontend_mod_scalar_int64_t,

        .pow_func        = ort_math_frontend_pow_int64_t,
        .pow_scalar_func = ort_math_frontend_pow_scalar_int64_t,
    },
    /* UINT8 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        .add_func        = ort_math_frontend_add_uint8_t,
        .add_scalar_func = ort_math_frontend_add_scalar_uint8_t,

        .sub_func        = ort_math_frontend_sub_uint8_t,
        .sub_scalar_func = ort_math_frontend_sub_scalar_uint8_t,
    
        .mul_func        = ort_math_frontend_mul_uint8_t,
        .mul_scalar_func = ort_math_frontend_mul_scalar_uint8_t,
    
        .div_func        = NULL,
        .div_scalar_func = NULL,
        .pow_func        = ort_math_frontend_pow_uint8_t,
        .ceil_func       = ort_math_frontend_ceil_uint8_t,
        .floor_func      = ort_math_frontend_floor_uint8_t,
        .round_func      = ort_math_frontend_round_uint8_t,
        .abs_func        = ort_math_frontend_abs_uint8_t,
        .sqrt_func       = NULL,
        .neg_func        = ort_math_frontend_neg_uint8_t,
        .recip_func      = NULL,
        .sign_func       = ort_math_frontend_sign_uint8_t,
        .trunc_func      = ort_math_frontend_trunc_uint8_t,
        .dot_func        = ort_math_frontend_dot_uint8_t,
        .matmul_func     = ort_math_frontend_matmul_uint8_t,
        .sum_func        = NULL,
        .sum_axis_func   = NULL,
        .min_func        = ort_math_frontend_min_uint8_t,
        .min_axis_func   = ort_math_frontend_min_axis_uint8_t,
        .max_func        = ort_math_frontend_max_uint8_t,
        .max_axis_func   = ort_math_frontend_max_axis_uint8_t,
        .mean_func       = NULL,
        .mean_axis_func  = NULL,
        .softmax_axis_func = NULL,

        .mod_func        = ort_math_frontend_mod_uint8_t,
        .mod_scalar_func = ort_math_frontend_mod_scalar_uint8_t,

        .pow_func        = ort_math_frontend_pow_uint8_t,
        .pow_scalar_func = ort_math_frontend_pow_scalar_uint8_t,
    },
    /* UINT16 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
        .add_func        = ort_math_frontend_add_uint16_t,
        .add_scalar_func = ort_math_frontend_add_scalar_uint16_t,

        .sub_func        = ort_math_frontend_sub_uint16_t,
        .sub_scalar_func = ort_math_frontend_sub_scalar_uint16_t,

        .mul_func        = ort_math_frontend_mul_uint16_t,
        .mul_scalar_func = ort_math_frontend_mul_scalar_uint16_t,

        .div_func        = NULL,
        .div_scalar_func = NULL,
        .ceil_func       = ort_math_frontend_ceil_uint16_t,
        .floor_func      = ort_math_frontend_floor_uint16_t,
        .round_func      = ort_math_frontend_round_uint16_t,
        .abs_func        = ort_math_frontend_abs_uint16_t,
        .sqrt_func       = NULL,
        .neg_func        = ort_math_frontend_neg_uint16_t,
        .recip_func      = NULL,
        .sign_func       = ort_math_frontend_sign_uint16_t,
        .trunc_func      = ort_math_frontend_trunc_uint16_t,
        .dot_func        = ort_math_frontend_dot_uint16_t,
        .matmul_func     = ort_math_frontend_matmul_uint16_t,
        .sum_func        = NULL,
        .sum_axis_func   = NULL,
        .min_func        = ort_math_frontend_min_uint16_t,
        .min_axis_func   = ort_math_frontend_min_axis_uint16_t,
        .max_func        = ort_math_frontend_max_uint16_t,
        .max_axis_func   = ort_math_frontend_max_axis_uint16_t,
        .mean_func       = NULL,
        .mean_axis_func  = NULL,
        .softmax_axis_func = NULL,

        .mod_func        = ort_math_frontend_mod_uint16_t,
        .mod_scalar_func = ort_math_frontend_mod_scalar_uint16_t,

        .pow_func        = ort_math_frontend_pow_uint16_t,
        .pow_scalar_func = ort_math_frontend_pow_scalar_uint16_t,
    },
    /* UINT32 */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
        .add_func        = ort_math_frontend_add_uint32_t,
        .add_scalar_func = ort_math_frontend_add_scalar_uint32_t,

        .sub_func        = ort_math_frontend_sub_uint32_t,
        .sub_scalar_func = ort_math_frontend_sub_scalar_uint32_t,

        .mul_func        = ort_math_frontend_mul_uint32_t,
        .mul_scalar_func = ort_math_frontend_mul_scalar_uint32_t,

        .div_func        = NULL,
        .div_scalar_func = NULL,
        .ceil_func       = ort_math_frontend_ceil_uint32_t,
        .floor_func      = ort_math_frontend_floor_uint32_t,
        .round_func      = ort_math_frontend_round_uint32_t,
        .abs_func        = ort_math_frontend_abs_uint32_t,
        .sqrt_func       = NULL,
        .neg_func        = ort_math_frontend_neg_uint32_t,
        .recip_func      = NULL,
        .sign_func       = ort_math_frontend_sign_uint32_t,
        .trunc_func      = ort_math_frontend_trunc_uint32_t,
        .dot_func        = ort_math_frontend_dot_uint32_t,
        .matmul_func     = ort_math_frontend_matmul_uint32_t,
        .sum_func        = NULL,
        .sum_axis_func   = NULL,
        .min_func        = ort_math_frontend_min_uint32_t,
        .min_axis_func   = ort_math_frontend_min_axis_uint32_t,
        .max_func        = ort_math_frontend_max_uint32_t,
        .max_axis_func   = ort_math_frontend_max_axis_uint32_t,
        .mean_func       = NULL,
        .mean_axis_func  = NULL,
        .softmax_axis_func = NULL,

        .mod_func        = ort_math_frontend_mod_uint32_t,
        .mod_scalar_func = ort_math_frontend_mod_scalar_uint32_t,

        .pow_func        = ort_math_frontend_pow_uint32_t,
        .pow_scalar_func = ort_math_frontend_pow_scalar_uint32_t,
    },
    /* BOOL */
    {
        .type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
        .add_func        = ort_math_frontend_add_zend_bool,          // Logical OR (or addition)
        .add_scalar_func = ort_math_frontend_add_scalar_zend_bool,   // Logical OR (or addition)

        .div_func        = NULL,          // Logical division (A && B)
        .div_scalar_func = NULL,          // Logical division (A && B)

        .sub_func        = NULL,                         // Logical XOR (or subtraction)
        .sub_scalar_func = NULL,                         // Logical XOR (or subtraction)
        .mul_func        = ort_math_frontend_mul_zend_bool,          // Logical AND (or multiplication)
        .mul_scalar_func = ort_math_frontend_mul_scalar_zend_bool,   // Logical AND (or multiplication)
        .ceil_func       = NULL,                         // Not meaningful for bool
        .floor_func      = NULL,                         // Not meaningful for bool
        .round_func      = NULL,                         // Not meaningful for bool
        .abs_func        = NULL,                         // Identity for bool
        .sqrt_func       = NULL,                         // Not meaningful for bool
        .neg_func        = ort_math_frontend_neg_zend_bool,         // Logical NOT
        .recip_func      = NULL,                         // Not meaningful for bool
        .sign_func       = NULL,                         // Identity for bool
        .trunc_func      = NULL,                         // Not meaningful for bool
        .dot_func        = NULL,
        .matmul_func     = NULL,
        .sum_func        = NULL,
        .sum_axis_func   = NULL,
        .min_func        = ort_math_frontend_min_zend_bool,
        .min_axis_func   = ort_math_frontend_min_axis_zend_bool,
        .max_func        = ort_math_frontend_max_zend_bool,
        .max_axis_func   = ort_math_frontend_max_axis_zend_bool,
        .mean_func       = NULL,
        .mean_axis_func  = NULL,
        .softmax_axis_func = NULL,

        .mod_func        = NULL,
        .mod_scalar_func = NULL,

        .pow_func        = NULL,                         // Logical power (A ** B)
        .pow_scalar_func = NULL,                         // Logical power (A ** B)  
    }
};

ort_math_dispatch_t* ort_math_dispatch_table(void) {
    return __ort_math_dispatch_table;
}

const ort_math_dispatch_t* ort_math_dispatch_type(
    ONNXTensorElementDataType type) {
    int16_t index = ort_math_dispatch_indexof(type);

    if (index == FAILURE) {
        memset(&__ort_math_dispatch_error,
            0, sizeof(ort_math_dispatch_t));
        return &__ort_math_dispatch_error;
    }

    return &__ort_math_dispatch_table[index];
}
