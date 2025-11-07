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

#ifndef HAVE_ORT_MATHS_TYPES
#define HAVE_ORT_MATHS_TYPES

typedef uint16_t float16;
typedef float    float32;
typedef double   float64;

/* IEEE 754 binary16 (FLOAT16) format constants */
#define ORT_MATH_F16_SIGN_MASK      0x8000  /* Sign bit mask */
#define ORT_MATH_F16_EXPONENT_MASK  0x7C00  /* Exponent field mask */
#define ORT_MATH_F16_MANTISSA_MASK  0x03FF  /* Mantissa field mask */
#define ORT_MATH_F16_EXPONENT_SHIFT 10      /* Exponent field bit position */
#define ORT_MATH_F16_EXPONENT_BIAS  15      /* Exponent bias */
#define ORT_MATH_F16_MAX_EXPONENT   31      /* Maximum exponent value (infinity/NaN) */
#define ORT_MATH_F16_MANTISSA_BITS  10      /* Number of mantissa bits */

/* IEEE 754 binary32 (FLOAT32) format constants */
#define ORT_MATH_F32_SIGN_SHIFT     31      /* Sign bit position */
#define ORT_MATH_F32_EXPONENT_MASK  0xFF    /* Exponent field mask (after shift) */
#define ORT_MATH_F32_EXPONENT_SHIFT 23      /* Exponent field bit position */
#define ORT_MATH_F32_EXPONENT_BIAS  127     /* Exponent bias */
#define ORT_MATH_F32_MAX_EXPONENT   255     /* Maximum exponent value (infinity/NaN) */
#define ORT_MATH_F32_MANTISSA_MASK  0x7FFFFF /* Mantissa field mask */
#define ORT_MATH_F32_MANTISSA_BITS  23      /* Number of mantissa bits */
#define ORT_MATH_F32_IMPLICIT_ONE   0x800000 /* Implicit leading 1 bit for normalized numbers */

/* Conversion constants */
#define ORT_MATH_F32_TO_F16_MANTISSA_SHIFT 13 /* Mantissa bits to discard (23-10) */
#define ORT_MATH_F32_TO_F16_ROUND_BIT      12 /* Bit position for rounding */
#define ORT_MATH_F32_TO_F16_STICKY_MASK    0xFFF /* Mask for sticky bits (bits 0-11) */
#define ORT_MATH_F16_DENORMAL_THRESHOLD    -10   /* Exponent threshold for denormal numbers */

/* Convert float32 to float16 */
static inline float16 ort_math_float16_from_float32(float32 value) {
    union { 
        float32 f;
        uint32_t  u; 
    } container = { .f = value };
    uint32_t bits = container.u;

    /* Extract IEEE 754 binary32 components */
    uint16_t sign = (bits >> 16) & ORT_MATH_F16_SIGN_MASK;
    uint32_t orig_exponent =
        (bits >> ORT_MATH_F32_EXPONENT_SHIFT) &
            ORT_MATH_F32_EXPONENT_MASK;
    int32_t exponent = (int32_t)
        orig_exponent -
            ORT_MATH_F32_EXPONENT_BIAS +
            ORT_MATH_F16_EXPONENT_BIAS;
    uint32_t mantissa =
        bits & ORT_MATH_F32_MANTISSA_MASK;

    /* Handle special cases */
    if (exponent <= 0) {
        /* Underflow or zero */
        if (exponent < ORT_MATH_F16_DENORMAL_THRESHOLD) {
            /* Too small for denormal, flush to zero */
            return sign;
        }
        /* Denormal number - add implicit bit and shift */
        mantissa = 
            (mantissa | ORT_MATH_F32_IMPLICIT_ONE) >>
                (1 - exponent);
        /* Note: could add rounding here for denormals */
        return sign | (mantissa >> ORT_MATH_F32_TO_F16_MANTISSA_SHIFT);
    }
    
    if (exponent >= ORT_MATH_F16_MAX_EXPONENT) {
        /* Overflow to infinity or NaN */
        if (orig_exponent == ORT_MATH_F32_MAX_EXPONENT && mantissa != 0) {
            /* NaN - preserve sign and ensure mantissa is non-zero */
            uint16_t nan_mantissa =
                (mantissa >> ORT_MATH_F32_TO_F16_MANTISSA_SHIFT);
            if (nan_mantissa == 0) {
                nan_mantissa = 1; /* Ensure it's a valid NaN */
            }
            return sign | ORT_MATH_F16_EXPONENT_MASK | nan_mantissa;
        }
        /* Infinity */
        return sign | ORT_MATH_F16_EXPONENT_MASK;
    }
    
    /* Normal number - round to nearest even */
    uint32_t round_bit =
        (mantissa >> ORT_MATH_F32_TO_F16_ROUND_BIT) & 1;
    uint32_t sticky_bits =
        mantissa & ORT_MATH_F32_TO_F16_STICKY_MASK;
    mantissa >>= ORT_MATH_F32_TO_F16_MANTISSA_SHIFT;
    
    /* Round to nearest, ties to even */
    if (round_bit && (sticky_bits > 0 || (mantissa & 1))) {
        mantissa++;
        if (mantissa > ORT_MATH_F16_MANTISSA_MASK) {
            /* Overflow in mantissa */
            exponent++;
            mantissa = 0;
            if (exponent >= ORT_MATH_F16_MAX_EXPONENT) {
                return sign | ORT_MATH_F16_EXPONENT_MASK;
            }
        }
    }
    
    return sign | (exponent << ORT_MATH_F16_EXPONENT_SHIFT) | mantissa;
}

/* Convert float16 to float32 */
static inline float32 ort_math_float32_from_float16(float16 value) {
    /* Extract IEEE 754 binary16 components */
    uint32_t sign = (value & ORT_MATH_F16_SIGN_MASK) << 16;
    uint32_t exponent =
        (value & ORT_MATH_F16_EXPONENT_MASK) >>
            ORT_MATH_F16_EXPONENT_SHIFT;
    uint32_t mantissa =
        (value & ORT_MATH_F16_MANTISSA_MASK) <<
            ORT_MATH_F32_TO_F16_MANTISSA_SHIFT;

    if (exponent == 0) {
        if (mantissa == 0) {
            /* Zero - sign preserved */
            union {
                uint32_t u;
                float32 f; 
            } container = {
                .u = sign 
            };
            return container.f;
        }
        /* Denormal - normalize it */
        exponent = 1;
        while ((mantissa & ORT_MATH_F32_IMPLICIT_ONE) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= ORT_MATH_F32_MANTISSA_MASK;
        exponent +=
            (ORT_MATH_F32_EXPONENT_BIAS -
                ORT_MATH_F16_EXPONENT_BIAS);
    } else if (exponent == ORT_MATH_F16_MAX_EXPONENT) {
        /* Infinity or NaN */
        exponent = ORT_MATH_F32_MAX_EXPONENT;
    } else {
        /* Normal number */
        exponent +=
            (ORT_MATH_F32_EXPONENT_BIAS -
                ORT_MATH_F16_EXPONENT_BIAS);
    }

    union {
        uint32_t u;
        float32 f; 
    } container = { 
        .u = sign |
            (exponent << ORT_MATH_F32_EXPONENT_SHIFT) | 
            mantissa 
    };

    return container.f;
}

/* Convert float16 to float64 */
static inline float64 ort_math_float64_from_float16(float16 value) {
    return (float64)ort_math_float32_from_float16(value);
}

/* Convert float64 to float16 */
static inline float16 ort_math_float16_from_float64(float64 value) {
    return ort_math_float16_from_float32((float32)value);
}

#endif