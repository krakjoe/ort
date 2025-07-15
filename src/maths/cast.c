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

#include "alloc.h"
#include "maths/cast.h"
#include "maths/pool.h"

#define ORT_MATH_CAST_SOURCE_CASE(SRC_TYPE, C_TYPE)             \
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_##SRC_TYPE: {            \
        C_TYPE src_val = *(C_TYPE*)src;                         \
        switch (dst_type) {                                     \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:           \
                *(float*)dst = (float)src_val; break;           \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:          \
                *(double*)dst = (double)src_val; break;         \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:            \
                *(int8_t*)dst = (int8_t)src_val; break;         \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:           \
                *(int16_t*)dst = (int16_t)src_val; break;       \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:           \
                *(int32_t*)dst = (int32_t)src_val; break;       \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:           \
                *(int64_t*)dst = (int64_t)src_val; break;       \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:           \
                *(uint8_t*)dst = (uint8_t)src_val; break;       \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:          \
                *(uint16_t*)dst = (uint16_t)src_val; break;     \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:          \
                *(uint32_t*)dst = (uint32_t)src_val; break;     \
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:            \
                *(uint8_t*)dst = (src_val != 0) ? 1 : 0; break; \
            default: break;                                     \
        }                                                       \
    } break;

void ort_math_cast_element(const void* src, void* dst, ONNXTensorElementDataType src_type, ONNXTensorElementDataType dst_type) {
    /* Fast path for same type */
    if (src_type == dst_type) {
        ort_memcpy(dst, src,
            php_ort_type_sizeof(src_type));
        return;
    }

    /* Handle type conversions using macros */
    switch (src_type) {
        ORT_MATH_CAST_SOURCE_CASE(FLOAT, float)
        ORT_MATH_CAST_SOURCE_CASE(DOUBLE, double)
        ORT_MATH_CAST_SOURCE_CASE(INT8, int8_t)
        ORT_MATH_CAST_SOURCE_CASE(INT16, int16_t)
        ORT_MATH_CAST_SOURCE_CASE(INT32, int32_t)
        ORT_MATH_CAST_SOURCE_CASE(INT64, int64_t)
        ORT_MATH_CAST_SOURCE_CASE(UINT8, uint8_t)
        ORT_MATH_CAST_SOURCE_CASE(UINT16, uint16_t)
        ORT_MATH_CAST_SOURCE_CASE(UINT32, uint32_t)
        
        /* Handle BOOL source type */
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            uint8_t src_val = *(uint8_t*)src;
            switch (dst_type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  
                    *(float*)dst = (float)src_val;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                    *(double*)dst = (double)src_val;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                    *(int8_t*)dst = (int8_t)src_val;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                    *(int16_t*)dst = (int16_t)src_val;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                    *(int32_t*)dst = (int32_t)src_val;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                    *(int64_t*)dst = (int64_t)src_val;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                    *(uint8_t*)dst = (uint8_t)src_val;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                    *(uint16_t*)dst = (uint16_t)src_val;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                    *(uint32_t*)dst = (uint32_t)src_val;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                    *(uint8_t*)dst = src_val;
                    break;
                default: break;
            }
        } break;
        
        default:
            break;
    }
}

void ort_math_cast_buffer(
    const void* src, void* dst,
    ONNXTensorElementDataType src_type,
    ONNXTensorElementDataType dst_type, size_t count) 
{
    /* Fast path for same type */
    if (src_type == dst_type) {
        size_t size =
            count *
                php_ort_type_sizeof(dst_type);
        ort_memcpy(dst, src, size);
        return;
    }

    /* Parallel path for large buffers on machines with many cores */
    size_t cores = ort_pool_cores();
    if (cores > 1 && count >= 256 * cores) {
        ort_pool_cast_ctx_t ctx = {
            .src      = src,
            .dst      = dst,
            .src_type = src_type,
            .dst_type = dst_type,
            .count    = count,
            .op       = ort_math_cast_element
        };

        size_t chunk;
        size_t num_chunks = ort_pool_chunk(
            count,
            php_ort_type_sizeof(dst_type), &chunk);

        ort_pool_submit(
            ort_pool_cast_worker, &ctx, num_chunks);
        return;
    }

    /* Serial slow path for small buffers */
    for (size_t i = 0; i < count; i++) {
        ort_math_cast_element(
            (const char*)src +
                i * php_ort_type_sizeof(src_type),
            (char*)dst +
                i * php_ort_type_sizeof(dst_type),
            src_type, dst_type);
    }
}