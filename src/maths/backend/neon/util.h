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

#include <arm_neon.h>

ORT_MATH_BACKEND_UTIL_DECL(
    hsum, float32x4, float, float32x4_t) {
    float32x2_t sum_pair = vadd_f32(
        vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(
        vpadd_f32(sum_pair, sum_pair), 0);
}

ORT_MATH_BACKEND_UTIL_DECL(
    hsum, float64x2, double, float64x2_t) {
    return vgetq_lane_f64(v, 0) +
           vgetq_lane_f64(v, 1);
}

ORT_MATH_BACKEND_UTIL_DECL(
    hsum, int32x4, int32_t, int32x4_t) {
    return vgetq_lane_s32(v, 0) +
           vgetq_lane_s32(v, 1) +
           vgetq_lane_s32(v, 2) +
           vgetq_lane_s32(v, 3);
}

ORT_MATH_BACKEND_UTIL_DECL(
    hsum, uint32x4, uint32_t, uint32x4_t) {
    return vgetq_lane_u32(v, 0) +
            vgetq_lane_u32(v, 1) + 
            vgetq_lane_u32(v, 2) +
            vgetq_lane_u32(v, 3);
}

ORT_MATH_BACKEND_UTIL_DECL(
    hsum, int16x8, int32_t, int16x8_t) {
    int32x2_t low = vpaddl_s16(vget_low_s16(v));
    int32x2_t high = vpaddl_s16(vget_high_s16(v));
    int32x4_t total = vcombine_s32(low, high);

    return ORT_MATH_BACKEND_UTIL(
        hsum, int32x4, int32_t)(total);
}

ORT_MATH_BACKEND_UTIL_DECL(
    hsum, uint16x8, uint32_t, uint16x8_t) {
    uint32x2_t low = vpaddl_u16(vget_low_u16(v));
    uint32x2_t high = vpaddl_u16(vget_high_u16(v));
    uint32x4_t total = vcombine_u32(low, high);

    return ORT_MATH_BACKEND_UTIL(
        hsum, uint32x4, uint32_t)(total);
}
