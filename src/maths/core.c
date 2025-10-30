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

#include "ort.h"

#if defined(ORT_BACKEND_CPU_ENABLED) || defined(ORT_BACKEND_GPU_ENABLED)
#include "maths/backend/impl.h"
#endif

#include "maths/pool.h"
#include "maths/promotion.h"

void ort_math_startup(void) {
#ifdef ORT_BACKEND_CPU_ENABLED
    ort_math_backend_cpu_install(
        ort_math_dispatch_table());
#endif
#ifdef ORT_BACKEND_GPU_ENABLED
    ort_math_backend_gpu_install(
        ort_math_dispatch_table());
#endif
    ort_math_promotion_startup();
}

void ort_math_activate(void) {
    ort_pool_init(
        ort_pool_cores());
}

void ort_math_deactivate(void) {
    ort_pool_destroy();
}

void ort_math_shutdown(void) {
    ort_math_promotion_shutdown();
}