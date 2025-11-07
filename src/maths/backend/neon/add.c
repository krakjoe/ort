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

#include "maths/backend/neon/impl.h"

#include <arm_neon.h>  /* NEON */

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, int8_t) {
    const int8_t* va = (const int8_t*)a;
    const int8_t* vb = (const int8_t*)b;
    int8_t* res = (int8_t*)result;
    const size_t mw = 16; // 16 int8_t per NEON register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_int8_fallback;
    }

    // Vectorized loop - process 16 int8_t at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        int8x16_t ma = vld1q_s8(&va[i]);
        int8x16_t mb = vld1q_s8(&vb[i]);
        int8x16_t mr = vaddq_s8(ma, mb);
        vst1q_s8(&res[i], mr);
    }

__ort_math_backend_add_int8_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, int8_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, int16_t) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;
    const size_t mw = 8; // 8 int16_t per NEON register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_int16_fallback;
    }

    // Vectorized loop - process 8 int16_t at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        int16x8_t ma = vld1q_s16(&va[i]);
        int16x8_t mb = vld1q_s16(&vb[i]);
        int16x8_t mr = vaddq_s16(ma, mb);
        vst1q_s16(&res[i], mr);
    }

__ort_math_backend_add_int16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, int16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, int32_t) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;
    const size_t mw = 4; // 4 int32_t per NEON register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_int32_fallback;
    }

    // Vectorized loop - process 4 int32_t at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        int32x4_t ma = vld1q_s32(&va[i]);
        int32x4_t mb = vld1q_s32(&vb[i]);
        int32x4_t mr = vaddq_s32(ma, mb);
        vst1q_s32(&res[i], mr);
    }

__ort_math_backend_add_int32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, int32_t)(
            res    + mc,
            va     + mc,
            vb     + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, int64_t) {
    const int64_t* va = (const int64_t*)a;
    const int64_t* vb = (const int64_t*)b;
    int64_t* res = (int64_t*)result;
    const size_t mw = 2; // 2 int64_t per NEON register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_int64_fallback;
    }

    // Vectorized loop - process 2 int64_t at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        int64x2_t ma = vld1q_s64(&va[i]);
        int64x2_t mb = vld1q_s64(&vb[i]);
        int64x2_t mr = vaddq_s64(ma, mb);
        vst1q_s64(&res[i], mr);
    }

__ort_math_backend_add_int64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, int64_t)(
            res    + mc,
            va     + mc,
            vb     + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, uint8_t) {
    const uint8_t* va = (const uint8_t*)a;
    const uint8_t* vb = (const uint8_t*)b;
    uint8_t* res = (uint8_t*)result;
    const size_t mw = 16; // 16 uint8_t per NEON register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_uint8_fallback;
    }

    // Vectorized loop - process 16 uint8_t at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        uint8x16_t ma = vld1q_u8(&va[i]);
        uint8x16_t mb = vld1q_u8(&vb[i]);
        uint8x16_t mr = vaddq_u8(ma, mb);
        vst1q_u8(&res[i], mr);
    }

__ort_math_backend_add_uint8_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, uint8_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, uint16_t) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;
    const size_t mw = 8; // 8 uint16_t per NEON register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_uint16_fallback;
    }

    // Vectorized loop - process 8 uint16_t at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        uint16x8_t ma = vld1q_u16(&va[i]);
        uint16x8_t mb = vld1q_u16(&vb[i]);
        uint16x8_t mr = vaddq_u16(ma, mb);
        vst1q_u16(&res[i], mr);
    }

__ort_math_backend_add_uint16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, uint16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, uint32_t) {
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;
    const size_t mw = 4; // 4 uint32_t per NEON register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_uint32_fallback;
    }

    // Vectorized loop - process 4 uint32_t at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        uint32x4_t ma = vld1q_u32(&va[i]);
        uint32x4_t mb = vld1q_u32(&vb[i]);
        uint32x4_t mr = vaddq_u32(ma, mb);
        vst1q_u32(&res[i], mr);
    }

__ort_math_backend_add_uint32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, uint32_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, float16) {
    const float16* va = (const float16*)a;
    const float16* vb = (const float16*)b;
    float16* res = (float16*)result;
    const size_t mw = 8; // 8 float16 per NEON register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_float16_fallback;
    }

    // Vectorized loop - process 8 float16 at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float16x8_t ma = vld1q_f16(&va[i]);
        float16x8_t mb = vld1q_f16(&vb[i]);
        float16x8_t mr = vaddq_f16(ma, mb);
        vst1q_f16(&res[i], mr);
    }

__ort_math_backend_add_float16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, float16)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, float32) {
    const float32* va = (const float32*)a;
    const float32* vb = (const float32*)b;
    float32* res = (float32*)result;
    const size_t mw = 4; // 4 float32 per NEON register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_float32_fallback;
    }

    // Vectorized loop - process 4 float32 at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float32x4_t ma = vld1q_f32(&va[i]);
        float32x4_t mb = vld1q_f32(&vb[i]);
        float32x4_t mr = vaddq_f32(ma, mb);
        vst1q_f32(&res[i], mr);
    }

__ort_math_backend_add_float32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, float32)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(neon, add, float64) {
    const float64* va = (const float64*)a;
    const float64* vb = (const float64*)b;
    float64* res = (float64*)result;
    const size_t mw = 2; // 2 float64 per NEON register

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_add_float64_fallback;
    }

    // Vectorized loop - process 2 float64 at once using NEON
    for (size_t i = 0; i < mc; i += mw) {
        float64x2_t ma = vld1q_f64(&va[i]);
        float64x2_t mb = vld1q_f64(&vb[i]);
        float64x2_t mr = vaddq_f64(ma, mb);
        vst1q_f64(&res[i], mr);
    }

__ort_math_backend_add_float64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(add, float64)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}