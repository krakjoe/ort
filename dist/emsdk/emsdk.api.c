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

#include <php.h>

#ifdef ZTS
#include <TSRM.h>
#endif

#ifdef EMSCRIPTEN
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include <SAPI.h>

#include <php_main.h>
#include <php_variables.h>
#include <zend_exceptions.h>

extern sapi_module_struct em_sapi_module;

extern void em_io_activate(void);

static const char EM_INI[] =
    "allow_url_fopen=1\n"
    "html_errors=0\n"
    "error_reporting=32767\n"
    "display_errors=1\n"
    "register_argc_argv=1\n"
    "implicit_flush=1\n"
    "output_buffering=0\n"
    "max_execution_time=0\n"
    "max_input_time=-1\n\0";

typedef void (*zend_error_func_t)(
    int type,
    zend_string *file,
    const uint32_t line,
    zend_string *message);
typedef void (*zend_log_func_t)(
    const char *message,
    int type);
typedef size_t (*zend_write_func_t)(
    const char* buf,
    size_t len);

static zend_write_func_t zend_write_func;
static zend_error_func_t zend_error_func;
static zend_log_func_t   zend_log_func;

typedef struct _em_string_t {
    char* value;
    size_t length;
    size_t max;
} em_string_t;

/* {{{ buffering */
em_string_t __em_buffer = (em_string_t) {
    .value  = NULL,
    .length = 0,
    .max    = 0,
};

void em_clear(bool _free) {
    if (__em_buffer.value) {
        if (_free) {
            free(__em_buffer.value);

            __em_buffer.value = NULL;
            __em_buffer.max   = 0;
        } else {
            memset(
                __em_buffer.value,
                0, __em_buffer.max);
        }
    } else {
        __em_buffer.max = 0;
    }
    __em_buffer.length = 0;
}

size_t em_buffer(const char* buf, size_t len) {
    if (__em_buffer.length + len >= __em_buffer.max) {
        __em_buffer.max = __em_buffer.length + len + 1024;
        __em_buffer.value = realloc(
            __em_buffer.value, __em_buffer.max);
        if (!__em_buffer.value) {
            return 0;
        }
    }

    memcpy(
    __em_buffer.value +
        __em_buffer.length,
    buf, len);
    __em_buffer.length += len;
    __em_buffer.value[
        __em_buffer.length] = 0;

    return len;
} /* }}} */

/* {{{ logging */
void em_buffer_log(const char* message, int type) {
    zend_string* msg = zend_strpprintf(
        0, "em internal error: %s",
        message
    );

    em_buffer(
        ZSTR_VAL(msg), ZSTR_LEN(msg));
    zend_string_release(msg);
}

void em_buffer_error(int type, zend_string* file, const uint32_t lineno, zend_string* message) {
    zend_string* msg = zend_strpprintf(
        0, "em error on line %u: %s",
        lineno,
        ZSTR_VAL(message)
    );
    em_buffer(ZSTR_VAL(msg), ZSTR_LEN(msg));
    zend_string_release(msg);
} /* }}} */

/* {{{ code lifecycle management */
static zend_always_inline zend_result
    em_activate(void) {
    if (php_request_startup() != SUCCESS) {
        php_module_shutdown();

        return FAILURE;
    }

    em_io_activate();

    em_clear(false);

    return SUCCESS;
}

static long em_string_read(void *handle, char *buf, size_t len) {
    em_string_t* string =
    	(em_string_t*)handle;
 
    if (len > string->length) {
    	len = string->length;
    }

    if (len == 0) {
        return len;
    }

    memcpy(
        buf, string->value, len);

    return len;
}

static size_t em_string_length(void *handle) {
    em_string_t* string =
    	(em_string_t*) handle;
    return string->length;
}

static void em_string_close(void *handle) { /* no op */ }

static zend_always_inline zend_op_array*
    em_compile(const char* code, size_t length) {
    em_string_t string =
    	(em_string_t) {
    	    .value   = (char*) code,
    	    .length  = length
    };

    // Create a file handle
    zend_file_handle fh;
    zend_stream_init_filename(
    	&fh, "virtual://script.php");
    fh.type = ZEND_HANDLE_STREAM;
    fh.handle.stream.handle = (void*)&string;
    fh.handle.stream.reader = em_string_read;
    fh.handle.stream.closer = em_string_close;
    fh.handle.stream.fsizer = em_string_length;
    fh.handle.stream.isatty = 0;

    zend_error_func = zend_error_cb;
    zend_error_cb   = em_buffer_error;

    zend_op_array* compiled = NULL;
    zend_try {
        compiled = zend_compile_file(
            &fh, ZEND_INCLUDE);
    } zend_end_try();

    if (EG(exception)) {
        zend_exception_error(
            EG(exception), E_COMPILE_ERROR);
    }

    zend_error_cb   = zend_error_func;

    zend_destroy_file_handle(&fh);

    return compiled;
}

static zend_always_inline
    void em_execute(zend_op_array* ops) {
    zval retval;
    ZVAL_UNDEF(&retval);

    zend_try {
        zend_execute(ops, &retval);
    } zend_end_try();

    if (EG(exception)) {
        zend_exception_error(EG(exception), E_ERROR);
    }

    destroy_op_array(ops);
    efree_size(ops, sizeof(zend_op_array));
}

static zend_always_inline void em_deactivate(void) {
    php_request_shutdown((void*) NULL);
} /* }}} */

/* {{{ exports */
int em_startup(void) {
#ifdef ZTS
    php_tsrm_startup();
#ifdef _WIN32
    ZEND_TSRMLS_CACHE_UPDATE();
#endif
#endif

    sapi_startup(&em_sapi_module);

    em_sapi_module.ini_entries = (char*) EM_INI;

  	if (em_sapi_module.startup(&em_sapi_module) == FAILURE) {
  		return FAILURE;
  	}

    SG(options)                |= SAPI_OPTION_NO_CHDIR;
    SG(headers_sent)            = 1;
    SG(request_info).no_headers = 1;

    zend_log_func   = sapi_module.log_message;
    zend_write_func = sapi_module.ub_write;

    sapi_module.ub_write = em_buffer;
    sapi_module.log_message = em_buffer_log;

    return SUCCESS;
}

uintptr_t EMSCRIPTEN_KEEPALIVE em_run_string(const char* code, size_t length) {
    if (em_activate() != SUCCESS) {
        return (uintptr_t) -1;
    }

    zend_op_array* ops =
        em_compile(code, length);

    if (ops) {
        em_execute(ops);
    }

    em_deactivate();

    return (uintptr_t) __em_buffer.value;
}

size_t EMSCRIPTEN_KEEPALIVE em_run_length(void) {
    return __em_buffer.length;
}

void EMSCRIPTEN_KEEPALIVE em_run_free(void) {
    em_clear(true);
}

void em_shutdown() {
    sapi_module.ub_write = zend_write_func;
    sapi_module.log_message = zend_log_func;

    php_module_shutdown();

    sapi_shutdown();

#ifdef ZTS
    tsrm_shutdown();
#endif
} /* }}} */

/* {{{ sapi gubbins */
static int em_sapi_startup(sapi_module_struct *sapi_module)
{
    return php_module_startup(sapi_module, NULL);
}

size_t em_sapi_write(const char* buf, size_t len) {
	const char *ptr = buf;
	size_t remaining = len;
	size_t ret;

	while (remaining > 0) {
        ret = fwrite(ptr, 1, MIN(remaining, 16384), stdout);
		if (!ret) {
			php_handle_aborted_connection();
		}
		ptr += ret;
		remaining -= ret;
	}

	return ret;
}

void em_sapi_error(int type, const char *message, ...) {
    va_list args;
    va_start(args, message);
    vfprintf(
        stderr,
        message,
        args);
    va_end(args);
}

void em_sapi_log(const char* message, int type) {
    fprintf(
        stderr,
        "em internal error: %s",
        message);
}

static void em_sapi_flush(void* ctx) {
    (void) ctx;

    if (fflush(stdout) == EOF ||
        fflush(stderr) == EOF) {
            php_handle_aborted_connection();
    }
}

static char* em_sapi_cookies(void)
{
	return NULL;
}

static void em_sapi_header(sapi_header_struct *header, void *ctx)
{
    (void) header;
    (void) ctx;
}

static void em_sapi_env(zval *vars)
{
	php_import_environment_variables(vars);
}

sapi_module_struct em_sapi_module = {
    "em",                         /* name */
    "em SAPI",                    /* pretty name */
    em_sapi_startup,              /* startup */
    php_module_shutdown_wrapper,  /* shutdown */
    NULL,                         /* activate */
    NULL,                         /* deactivate */
    em_sapi_write,                /* ub write */
    em_sapi_flush,                /* flush */
    NULL,                         /* get uid */
    NULL,                         /* getenv */
    em_sapi_error,                /* error handler */
    NULL,                         /* header handler */
    NULL,                         /* send headers handler */
    em_sapi_header,               /* send header handler */
    NULL,                         /* read post */
    em_sapi_cookies,              /* read cookies */
    em_sapi_env,                  /* register server variables */
    em_sapi_log,                  /* log message */
    NULL,                         /* get request time */
    NULL,                         /* terminate process */
    STANDARD_SAPI_MODULE_PROPERTIES
};
/* }}} */