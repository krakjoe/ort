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

#include "env.h"
#include "alloc.h"
#include "status.h"
#include "tensor.h"
#include "generators.h"
#include "maths.h"

#ifdef HAVE_ONNXRUNTIME
#include "options.h"
#include "runtime.h"
#include "model.h"
#include "iterator.h"

const OrtApi* api;
#endif

PHP_MINIT_FUNCTION(ORT_CORE)
{
#ifdef HAVE_ONNXRUNTIME
    api = OrtGetApiBase()
            ->GetApi(ORT_API_VERSION);
#endif

    PHP_MINIT(ORT_ENV)(INIT_FUNC_ARGS_PASSTHRU);

    ort_alloc_startup();

    PHP_MINIT(ORT_STATUS)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_TENSOR)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_GENERATORS)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_MATH)(INIT_FUNC_ARGS_PASSTHRU);

#ifdef HAVE_ONNXRUNTIME
    PHP_MINIT(ORT_OPTIONS)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_RUNTIME)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_MODEL)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MINIT(ORT_ITERATOR)(INIT_FUNC_ARGS_PASSTHRU);
#endif

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_CORE)
{
#ifdef HAVE_ONNXRUNTIME
    PHP_MSHUTDOWN(ORT_ITERATOR)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_MODEL)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_RUNTIME)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_OPTIONS)(INIT_FUNC_ARGS_PASSTHRU);
#endif

    PHP_MSHUTDOWN(ORT_MATH)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_GENERATORS)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_TENSOR)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_MSHUTDOWN(ORT_STATUS)(INIT_FUNC_ARGS_PASSTHRU);

    ort_alloc_shutdown();

    PHP_MSHUTDOWN(ORT_ENV)(INIT_FUNC_ARGS_PASSTHRU);

    return SUCCESS;
}

PHP_RINIT_FUNCTION(ORT_CORE)
{
    ort_alloc_activate();

#ifdef HAVE_ONNXRUNTIME
    PHP_RINIT(ORT_TENSOR)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_RINIT(ORT_MODEL)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_RINIT(ORT_RUNTIME)(INIT_FUNC_ARGS_PASSTHRU);
#endif

    PHP_RINIT(ORT_MATH)(INIT_FUNC_ARGS_PASSTHRU);

    return SUCCESS;
}

PHP_RSHUTDOWN_FUNCTION(ORT_CORE)
{
    PHP_RSHUTDOWN(ORT_MATH)(INIT_FUNC_ARGS_PASSTHRU);

#ifdef HAVE_ONNXRUNTIME
    PHP_RSHUTDOWN(ORT_RUNTIME)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_RSHUTDOWN(ORT_MODEL)(INIT_FUNC_ARGS_PASSTHRU);
    PHP_RSHUTDOWN(ORT_TENSOR)(INIT_FUNC_ARGS_PASSTHRU);
#endif

    ort_alloc_deactivate();

    return SUCCESS;
}
