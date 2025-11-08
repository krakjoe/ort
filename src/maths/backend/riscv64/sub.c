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

#include "maths/backend/riscv64/impl.h"
#include <riscv_vector.h> /* RVV */

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, int8_t) {
    const int8_t* va = (const int8_t*)a;
    const int8_t* vb = (const int8_t*)b;
    int8_t* res = (int8_t*)result;
    const size_t mw = __riscv_vsetvlmax_e8m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_int8_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        vint8m1_t ma = __riscv_vle8_v_i8m1(&va[i], mw);
        vint8m1_t mb = __riscv_vle8_v_i8m1(&vb[i], mw);
        vint8m1_t mr = __riscv_vsub_vv_i8m1(ma, mb, mw);
        __riscv_vse8_v_i8m1(&res[i], mr, mw);
    }

__ort_math_backend_sub_int8_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, int8_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, int16_t) {
    const int16_t* va = (const int16_t*)a;
    const int16_t* vb = (const int16_t*)b;
    int16_t* res = (int16_t*)result;
    const size_t mw = __riscv_vsetvlmax_e16m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_int16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        vint16m1_t ma = __riscv_vle16_v_i16m1(&va[i], mw);
        vint16m1_t mb = __riscv_vle16_v_i16m1(&vb[i], mw);
        vint16m1_t mr = __riscv_vsub_vv_i16m1(ma, mb, mw);
        __riscv_vse16_v_i16m1(&res[i], mr, mw);
    }

__ort_math_backend_sub_int16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, int16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, int32_t) {
    const int32_t* va = (const int32_t*)a;
    const int32_t* vb = (const int32_t*)b;
    int32_t* res = (int32_t*)result;
    const size_t mw = __riscv_vsetvlmax_e32m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_int32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        vint32m1_t ma = __riscv_vle32_v_i32m1(&va[i], mw);
        vint32m1_t mb = __riscv_vle32_v_i32m1(&vb[i], mw);
        vint32m1_t mr = __riscv_vsub_vv_i32m1(ma, mb, mw);
        __riscv_vse32_v_i32m1(&res[i], mr, mw);
    }

__ort_math_backend_sub_int32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, int32_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, int64_t) {
    const int64_t* va = (const int64_t*)a;
    const int64_t* vb = (const int64_t*)b;
    int64_t* res = (int64_t*)result;
    const size_t mw = __riscv_vsetvlmax_e64m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_int64_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        vint64m1_t ma = __riscv_vle64_v_i64m1(&va[i], mw);
        vint64m1_t mb = __riscv_vle64_v_i64m1(&vb[i], mw);
        vint64m1_t mr = __riscv_vsub_vv_i64m1(ma, mb, mw);
        __riscv_vse64_v_i64m1(&res[i], mr, mw);
    }

__ort_math_backend_sub_int64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, int64_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, uint8_t) {
    const uint8_t* va = (const uint8_t*)a;
    const uint8_t* vb = (const uint8_t*)b;
    uint8_t* res = (uint8_t*)result;
    const size_t mw = __riscv_vsetvlmax_e8m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_uint8_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        vuint8m1_t ma = __riscv_vle8_v_u8m1(&va[i], mw);
        vuint8m1_t mb = __riscv_vle8_v_u8m1(&vb[i], mw);
        vuint8m1_t mr = __riscv_vsub_vv_u8m1(ma, mb, mw);
        __riscv_vse8_v_u8m1(&res[i], mr, mw);
    }

__ort_math_backend_sub_uint8_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, uint8_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, uint16_t) {
    const uint16_t* va = (const uint16_t*)a;
    const uint16_t* vb = (const uint16_t*)b;
    uint16_t* res = (uint16_t*)result;
    const size_t mw = __riscv_vsetvlmax_e16m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_uint16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        vuint16m1_t ma = __riscv_vle16_v_u16m1(&va[i], mw);
        vuint16m1_t mb = __riscv_vle16_v_u16m1(&vb[i], mw);
        vuint16m1_t mr = __riscv_vsub_vv_u16m1(ma, mb, mw);
        __riscv_vse16_v_u16m1(&res[i], mr, mw);
    }

__ort_math_backend_sub_uint16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, uint16_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, uint32_t) {
    const uint32_t* va = (const uint32_t*)a;
    const uint32_t* vb = (const uint32_t*)b;
    uint32_t* res = (uint32_t*)result;
    const size_t mw = __riscv_vsetvlmax_e32m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_uint32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        vuint32m1_t ma = __riscv_vle32_v_u32m1(&va[i], mw);
        vuint32m1_t mb = __riscv_vle32_v_u32m1(&vb[i], mw);
        vuint32m1_t mr = __riscv_vsub_vv_u32m1(ma, mb, mw);
        __riscv_vse32_v_u32m1(&res[i], mr, mw);
    }

__ort_math_backend_sub_uint32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, uint32_t)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

#ifdef ORT_BACKEND_CPU_F16V
ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, float16) {
    const float16* va = (const float16*)a;
    const float16* vb = (const float16*)b;
    float16* res = (float16*)result;
    const size_t mw = __riscv_vsetvlmax_e16m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_float16_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        vfloat16m1_t ma =
            __riscv_vle16_v_f16m1(
                (const _Float16*)&va[i], mw);
        vfloat16m1_t mb =
            __riscv_vle16_v_f16m1(
                (const _Float16*)&vb[i], mw);
        vfloat16m1_t mr = __riscv_vfsub_vv_f16m1(ma, mb, mw);
        __riscv_vse16_v_f16m1(
            (_Float16*)&res[i], mr, mw);
    }

__ort_math_backend_sub_float16_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, float16)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}
#endif

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, float32) {
    const float32* va = (const float32*)a;
    const float32* vb = (const float32*)b;
    float32* res = (float32*)result;
    const size_t mw = __riscv_vsetvlmax_e32m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_float32_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        vfloat32m1_t ma = __riscv_vle32_v_f32m1(&va[i], mw);
        vfloat32m1_t mb = __riscv_vle32_v_f32m1(&vb[i], mw);
        vfloat32m1_t mr = __riscv_vfsub_vv_f32m1(ma, mb, mw);
        __riscv_vse32_v_f32m1(&res[i], mr, mw);
    }

__ort_math_backend_sub_float32_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, float32)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}

ORT_MATH_BACKEND_BINARY_OP_DECL(riscv64, sub, float64) {
    const float64* va = (const float64*)a;
    const float64* vb = (const float64*)b;
    float64* res = (float64*)result;
    const size_t mw = __riscv_vsetvlmax_e64m1();

    size_t mc = ort_math_backend_optimal_count(count, mw);

    if (mc == 0) {
        goto __ort_math_backend_sub_float64_fallback;
    }

    for (size_t i = 0; i < mc; i += mw) {
        vfloat64m1_t ma = __riscv_vle64_v_f64m1(&va[i], mw);
        vfloat64m1_t mb = __riscv_vle64_v_f64m1(&vb[i], mw);
        vfloat64m1_t mr = __riscv_vfsub_vv_f64m1(ma, mb, mw);
        __riscv_vse64_v_f64m1(&res[i], mr, mw);
    }

__ort_math_backend_sub_float64_fallback:
    if (mc < count) {
        ORT_MATH_FRONTEND_OP_SYMBOL(sub, float64)(
            res   + mc,
            va    + mc,
            vb    + mc,
            count - mc);
    }
}