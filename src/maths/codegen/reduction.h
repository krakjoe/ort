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

/**
 * @file codegen/reduction.h
 * @brief Reduction kernel code generation macros
 */

#define ORT_MATH_REDUCTION_KERNEL_AXIS_PROLOGUE() \
    size_t outer = 1, inner = 1; \
    for (size_t i = 0; i < axis; ++i) { \
        outer *= input_shape[i]; \
    } \
    for (size_t i = axis + 1; i < input_dims; ++i) { \
        inner *= input_shape[i]; \
    } \
    for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) { \
        for (size_t inner_idx = 0; inner_idx < inner; ++inner_idx) { \
            int64_t indices[ORT_MATH_RESULT_STACK_DIMENSIONS]; \
            for (size_t d = 0; d < input_dims; ++d) { \
                indices[d] = 0; \
            } \
            size_t tmp_outer = outer_idx; \
            for (size_t d = 0; d < axis; ++d) { \
                indices[d] = tmp_outer % input_shape[d]; \
                tmp_outer /= input_shape[d]; \
            } \
            size_t tmp_inner = inner_idx; \
            for (size_t d = input_dims - 1; d > axis; --d) { \
                indices[d] = tmp_inner % input_shape[d]; \
                tmp_inner /= input_shape[d]; \
            }

#define ORT_MATH_REDUCTION_KERNEL_AXIS_EPILOGUE() \
    int64_t out_indices[ORT_MATH_RESULT_STACK_DIMENSIONS]; \
    if (output_dims == input_dims) { /* keepdims=true */ \
        size_t j = 0; \
        for (size_t d = 0; d < input_dims; ++d) { \
            if (d == axis) { \
                out_indices[j++] = 0; \
            } else { \
                out_indices[j++] = indices[d]; \
            } \
        } \
    } else { /* keepdims=false */ \
        size_t j = 0; \
        for (size_t d = 0; d < input_dims; ++d) { \
            if (d != axis) { \
                out_indices[j++] = indices[d]; \
            } \
        } \
    }

#define ORT_MATH_REDUCTION_KERNEL_AXIS_LEAVE() \
    } \
}
