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

#include "php.h"
#include "ext/standard/info.h"

#include "ort.h"
#include "maths/pool.h"

#include "php_ort.h"

PHP_MINIT_FUNCTION(ort)
{
	PHP_MINIT(ORT_CORE)(INIT_FUNC_ARGS_PASSTHRU);

	return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ort)
{
	PHP_MSHUTDOWN(ORT_CORE)(INIT_FUNC_ARGS_PASSTHRU);

	return SUCCESS;
}

PHP_RINIT_FUNCTION(ort)
{
#if defined(ZTS) && defined(COMPILE_DL_ORT)
	ZEND_TSRMLS_CACHE_UPDATE();
#endif
	PHP_RINIT(ORT_CORE)(INIT_FUNC_ARGS_PASSTHRU);

	return SUCCESS;
}

PHP_RSHUTDOWN_FUNCTION(ort)
{
	PHP_RSHUTDOWN(ORT_CORE)(INIT_FUNC_ARGS_PASSTHRU);
	return SUCCESS;
}

PHP_MINFO_FUNCTION(ort)
{
	php_info_print_table_start();
#ifdef HAVE_ONNXRUNTIME
	php_info_print_table_row(2, "ONNX Runtime support", "enabled");
#else
	php_info_print_table_row(2, "ONNX Runtime support", "disabled");
#endif
#ifdef HAVE_ORT_POOL
	{
		char buffer[256];

		snprintf(buffer,
			sizeof(buffer),
			"%zu", ort_pool_cores());
		php_info_print_table_row(2, "Thread Pool", buffer);
	}
#else
	php_info_print_table_row(2, "Thread Pool", "disabled");
#endif
#ifdef ORT_BACKEND_CPU_ENABLED
	php_info_print_table_row(2, "CPU Backend", ORT_BACKEND_CPU_NAME);
#else
	php_info_print_table_row(2, "CPU Backend", "none");
#endif
#ifdef ORT_BACKEND_GPU_ENABLED
	php_info_print_table_row(2, "GPU Backend", ORT_BACKEND_GPU_NAME);
#else
	php_info_print_table_row(2, "GPU Backend", "none");
#endif

	php_info_print_table_end();
}

zend_module_entry ort_module_entry = {
	STANDARD_MODULE_HEADER,
	"ort",				/* Extension name */
	NULL,				/* zend_function_entry */
	PHP_MINIT(ort),			/* PHP_MINIT - Module initialization */
	PHP_MSHUTDOWN(ort),		/* PHP_MSHUTDOWN - Module shutdown */
	PHP_RINIT(ort),			/* PHP_RINIT - Request initialization */
	PHP_RSHUTDOWN(ort),		/* PHP_RSHUTDOWN - Request shutdown */
	PHP_MINFO(ort),			/* PHP_MINFO - Module info */
	PHP_ORT_VERSION,		/* Version */
	STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_ORT
# ifdef ZTS
ZEND_TSRMLS_CACHE_DEFINE()
# endif
ZEND_GET_MODULE(ort)
#endif
