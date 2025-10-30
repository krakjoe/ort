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

#include <cuda_runtime.h>

void ort_cuda_matmul_int8(int8_t* result, const int8_t* a, size_t a_cols, 
                          const int8_t* b, size_t b_cols, cudaStream_t stream);
void ort_cuda_matmul_uint8(uint8_t* result, const uint8_t* a, size_t a_cols, 
                           const uint8_t* b, size_t b_cols, cudaStream_t stream);
void ort_cuda_matmul_int16(int16_t* result, const int16_t* a, size_t a_cols, 
                           const int16_t* b, size_t b_cols, cudaStream_t stream);
void ort_cuda_matmul_uint16(uint16_t* result, const uint16_t* a, size_t a_cols, 
                            const uint16_t* b, size_t b_cols, cudaStream_t stream);
void ort_cuda_matmul_int32(int32_t* result, const int32_t* a, size_t a_cols, 
                           const int32_t* b, size_t b_cols, cudaStream_t stream);
void ort_cuda_matmul_uint32(uint32_t* result, const uint32_t* a, size_t a_cols, 
                            const uint32_t* b, size_t b_cols, cudaStream_t stream);