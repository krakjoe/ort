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

#include "maths/backend/impl.h"

void ort_math_backend_install(ort_math_dispatch_t* table) {
    /* abs.c */
    ORT_MATH_DISPATCH_INSTALL(table, FLOAT,   abs, ort_math_backend_abs_float);
    ORT_MATH_DISPATCH_INSTALL(table, DOUBLE,  abs, ort_math_backend_abs_double);

    /* add.c */
    ORT_MATH_DISPATCH_INSTALL(table, FLOAT,   add, ort_math_backend_add_float);
    ORT_MATH_DISPATCH_INSTALL(table, DOUBLE,  add, ort_math_backend_add_double);
    ORT_MATH_DISPATCH_INSTALL(table, INT8,    add, ort_math_backend_add_int8_t);
    ORT_MATH_DISPATCH_INSTALL(table, INT16,   add, ort_math_backend_add_int16_t);
    ORT_MATH_DISPATCH_INSTALL(table, INT32,   add, ort_math_backend_add_int32_t);
    ORT_MATH_DISPATCH_INSTALL(table, UINT8,   add, ort_math_backend_add_uint8_t);
    ORT_MATH_DISPATCH_INSTALL(table, UINT16,  add, ort_math_backend_add_uint16_t);
    ORT_MATH_DISPATCH_INSTALL(table, UINT32,  add, ort_math_backend_add_uint32_t);

    /* div.c */
    ORT_MATH_DISPATCH_INSTALL(table, FLOAT,   div, ort_math_backend_div_float);
    ORT_MATH_DISPATCH_INSTALL(table, DOUBLE,  div, ort_math_backend_div_double);

    /* matmul.c */
    ORT_MATH_DISPATCH_INSTALL(table, FLOAT,   matmul, ort_math_backend_matmul_float);
    ORT_MATH_DISPATCH_INSTALL(table, DOUBLE,  matmul, ort_math_backend_matmul_double);
    ORT_MATH_DISPATCH_INSTALL(table, INT16,   matmul, ort_math_backend_matmul_int16_t);
    ORT_MATH_DISPATCH_INSTALL(table, UINT16,  matmul, ort_math_backend_matmul_uint16_t);

    /* mul.c */
    ORT_MATH_DISPATCH_INSTALL(table, FLOAT,   mul, ort_math_backend_mul_float);
    ORT_MATH_DISPATCH_INSTALL(table, DOUBLE,  mul, ort_math_backend_mul_double);
    ORT_MATH_DISPATCH_INSTALL(table, INT16,   mul, ort_math_backend_mul_int16_t);
    ORT_MATH_DISPATCH_INSTALL(table, UINT16,  mul, ort_math_backend_mul_uint16_t);

    /* neg.c */
    ORT_MATH_DISPATCH_INSTALL(table, FLOAT,   neg, ort_math_backend_neg_float);
    ORT_MATH_DISPATCH_INSTALL(table, DOUBLE,  neg, ort_math_backend_neg_double);

    /* recip.c */
    ORT_MATH_DISPATCH_INSTALL(table, FLOAT,   recip, ort_math_backend_recip_float);
    ORT_MATH_DISPATCH_INSTALL(table, DOUBLE,  recip, ort_math_backend_recip_double);

    /* sign.c */
    ORT_MATH_DISPATCH_INSTALL(table, FLOAT,   sign, ort_math_backend_sign_float);
    ORT_MATH_DISPATCH_INSTALL(table, DOUBLE,  sign, ort_math_backend_sign_double);

    /* sqrt.c */
    ORT_MATH_DISPATCH_INSTALL(table, FLOAT,   sqrt, ort_math_backend_sqrt_float);
    ORT_MATH_DISPATCH_INSTALL(table, DOUBLE,  sqrt, ort_math_backend_sqrt_double);

    /* sub.c */
    ORT_MATH_DISPATCH_INSTALL(table, FLOAT,   sub, ort_math_backend_sub_float);
    ORT_MATH_DISPATCH_INSTALL(table, DOUBLE,  sub, ort_math_backend_sub_double);
    ORT_MATH_DISPATCH_INSTALL(table, INT8,    sub, ort_math_backend_sub_int8_t);
    ORT_MATH_DISPATCH_INSTALL(table, INT16,   sub, ort_math_backend_sub_int16_t);
    ORT_MATH_DISPATCH_INSTALL(table, INT32,   sub, ort_math_backend_sub_int32_t);
    ORT_MATH_DISPATCH_INSTALL(table, INT64,   sub, ort_math_backend_sub_int64_t);
    ORT_MATH_DISPATCH_INSTALL(table, UINT8,   sub, ort_math_backend_sub_uint8_t);
    ORT_MATH_DISPATCH_INSTALL(table, UINT16,  sub, ort_math_backend_sub_uint16_t);
    ORT_MATH_DISPATCH_INSTALL(table, UINT32,  sub, ort_math_backend_sub_uint32_t);
}
