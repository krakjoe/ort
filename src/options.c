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

#include "options.h"
#include "status.h"

zend_class_entry* php_ort_options_ce;
zend_object_handlers php_ort_options_handlers;

ort_options_t* __php_ort_options_default = NULL;

ort_options_t* php_ort_options_default() {
    return __php_ort_options_default;
}

static zend_always_inline void php_ort_options_free(ort_options_t* ort) {
    zend_bool persistent = 
        (ort->owner == PHP_ORT_OWN_HEAP) ? 
            1 : 0;

    if (ort->options) {
        api->ReleaseSessionOptions(ort->options);
    }

    pefree(ort, persistent);
}

void php_ort_options_release(ort_options_t *ort) {
    if (ort == __php_ort_options_default) {
        return;
    }

    if (php_ort_atomic_delref(&ort->refcount) == 0) {
        php_ort_options_free(ort);
    }
}

static zend_object* php_ort_options_create(zend_class_entry *type) {
    php_ort_options_t* ort = ecalloc(1, 
        sizeof(php_ort_options_t) + zend_object_properties_size(type));

    zend_object_std_init(&ort->std, php_ort_options_ce);

    ort->std.handlers = &php_ort_options_handlers;

    ort->object = pecalloc(1, sizeof(ort_options_t), 0);
    ort->object->refcount = 1;
    ort->object->owner    = PHP_ORT_OWN_ZEND;

    return &ort->std;
}

static void php_ort_options_destroy(zend_object* zo) {
    php_ort_options_t* ort =
        php_ort_options_fetch(zo);

    php_ort_options_release(ort->object);

    zend_object_std_dtor(zo);
}

ZEND_BEGIN_ARG_INFO_EX(php_ort_options_construct_arginfo, 0, 0, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Options, __construct)
{
    php_ort_options_t* ort =
        php_ort_options_fetch(Z_OBJ(EX(This)));
    OrtStatus* status;

    ZEND_PARSE_PARAMETERS_NONE();

    php_ort_status_flow(
        (status = api->CreateSessionOptions(
            &ort->object->options)),
        {
            api->ReleaseStatus(status);

            return;
        },
        php_ort_status_error_ce,
        "failed to allocate options: %s",
        api->GetErrorMessage(status)
    );
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_options_setExecutionMode_arginfo, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, mode, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Options, setExecutionMode)
{
    php_ort_options_t* ort =
        php_ort_options_fetch(Z_OBJ(EX(This)));
    zend_long mode;
    OrtStatus* status;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(mode)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_status_flow(
        (status = api->SetSessionExecutionMode(
            ort->object->options, mode)),
        {
            api->ReleaseStatus(status);

            return;
        },
        php_ort_status_error_ce,
        "failed to set execution mode: %s",
        api->GetErrorMessage(status)
    );
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_options_setGraphOptimization_arginfo, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, optimization, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Options, setGraphOptimization)
{
    php_ort_options_t* ort =
        php_ort_options_fetch(Z_OBJ(EX(This)));
    zend_long optimization;
    OrtStatus* status;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(optimization)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_status_flow(
        (status = api->SetSessionGraphOptimizationLevel(
            ort->object->options, optimization)),
        {
            api->ReleaseStatus(status);

            return;
        },
        php_ort_status_error_ce,
        "failed to set graph optimization: %s",
        api->GetErrorMessage(status)
    );
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_options_setLogId_arginfo, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, id, IS_STRING, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Options, setLogId)
{
    php_ort_options_t* ort =
        php_ort_options_fetch(Z_OBJ(EX(This)));
    zend_string* id;
    OrtStatus* status;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(id)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_status_flow(
        (status = api->SetSessionLogId(
            ort->object->options, ZSTR_VAL(id))),
        {
            api->ReleaseStatus(status);

            return;
        },
        php_ort_status_error_ce,
        "failed to set log id (%s): %s",
        ZSTR_VAL(id),
        api->GetErrorMessage(status)
    );
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_options_setLogSeverity_arginfo, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, severity, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Options, setLogSeverity)
{
    php_ort_options_t* ort =
        php_ort_options_fetch(Z_OBJ(EX(This)));
    zend_long severity;
    OrtStatus* status;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(severity)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_status_flow(
        (status = api->SetSessionLogSeverityLevel(
            ort->object->options, severity)),
        {
            api->ReleaseStatus(status);

            return;
        },
        php_ort_status_error_ce,
        "failed to set log severity: %s",
        api->GetErrorMessage(status)
    );
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_options_setLogVerbosity_arginfo, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, verbosity, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Options, setLogVerbosity)
{
    php_ort_options_t* ort =
        php_ort_options_fetch(Z_OBJ(EX(This)));
    zend_long verbosity;
    OrtStatus* status;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(verbosity)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_status_flow(
        (status = api->SetSessionLogVerbosityLevel(
            ort->object->options, verbosity)),
        {
            api->ReleaseStatus(status);

            return;
        },
        php_ort_status_error_ce,
        "failed to set log verbosity: %s",
        api->GetErrorMessage(status)
    );
}

zend_function_entry php_ort_options_methods[] = {
    PHP_ME(ONNX_Options, __construct,          php_ort_options_construct_arginfo,            ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Options, setExecutionMode,     php_ort_options_setExecutionMode_arginfo,     ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Options, setGraphOptimization, php_ort_options_setGraphOptimization_arginfo, ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Options, setLogId,             php_ort_options_setLogId_arginfo,             ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Options, setLogSeverity,       php_ort_options_setLogSeverity_arginfo,       ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Options, setLogVerbosity,      php_ort_options_setLogVerbosity_arginfo,      ZEND_ACC_PUBLIC)

    PHP_FE_END
};

PHP_MINIT_FUNCTION(ORT_OPTIONS)
{
    zend_class_entry ce;

    INIT_NS_CLASS_ENTRY(ce, "ORT", "Options", php_ort_options_methods);

    php_ort_options_ce = zend_register_internal_class(&ce);
    php_ort_options_ce->create_object = php_ort_options_create;

#ifdef ZEND_ACC_NOT_SERIALIZABLE
    php_ort_options_ce->ce_flags |= ZEND_ACC_NOT_SERIALIZABLE;
  #else
    php_ort_options_ce->serialize = zend_class_serialize_deny;
    php_ort_options_ce->unserialize = zend_class_unserialize_deny;
#endif

    memcpy(&php_ort_options_handlers,
        zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    php_ort_options_handlers.offset    = XtOffsetOf(php_ort_options_t, std);
    php_ort_options_handlers.free_obj  = php_ort_options_destroy;
    php_ort_options_handlers.clone_obj = NULL;

    /* Log Verbosity/Severity Flags (Prefix L) */
    zend_declare_class_constant_long(
        php_ort_options_ce,
        "LVERBOSE", sizeof("LVERBOSE")-1,
        ORT_LOGGING_LEVEL_VERBOSE);

    zend_declare_class_constant_long(
        php_ort_options_ce,
        "LINFO", sizeof("LINFO")-1,
        ORT_LOGGING_LEVEL_INFO);

    zend_declare_class_constant_long(
        php_ort_options_ce,
        "LWARNING", sizeof("LWARNING")-1,
        ORT_LOGGING_LEVEL_WARNING);

    zend_declare_class_constant_long(
        php_ort_options_ce,
        "LERROR", sizeof("LERROR")-1,
        ORT_LOGGING_LEVEL_ERROR);

    zend_declare_class_constant_long(
        php_ort_options_ce,
        "LFATAL", sizeof("LFATAL")-1,
        ORT_LOGGING_LEVEL_FATAL);

    /* Execution Mode Flags (Prefix E) */
    zend_declare_class_constant_long(
        php_ort_options_ce,
        "ESEQUENTIAL", sizeof("ESEQUENTIAL")-1,
        ORT_SEQUENTIAL);

    zend_declare_class_constant_long(
        php_ort_options_ce,
        "EPARALLEL", sizeof("EPARALLEL")-1,
        ORT_PARALLEL);

    /* Optimization Flags (Prefix O) */
    zend_declare_class_constant_long(
        php_ort_options_ce,
        "ONONE", sizeof("ONONE")-1,
        ORT_DISABLE_ALL);

    zend_declare_class_constant_long(
        php_ort_options_ce,
        "OBASIC", sizeof("OBASIC")-1,
        ORT_ENABLE_BASIC);

    zend_declare_class_constant_long(
        php_ort_options_ce,
        "OEXTENDED", sizeof("OEXTENDED")-1,
        ORT_ENABLE_EXTENDED);

    zend_declare_class_constant_long(
        php_ort_options_ce,
        "OALL", sizeof("OALL")-1,
        ORT_ENABLE_ALL);

    OrtStatus* status;

    __php_ort_options_default = pecalloc(1,
        sizeof(ort_options_t), 1);
    __php_ort_options_default->refcount = 1;
    __php_ort_options_default->owner    = PHP_ORT_OWN_HEAP;

    /** 
     * We don't care about coverage here, these calls must not fail anyway
     * 
     * LCOV_EXCL_START 
     **/
    status = api->CreateSessionOptions(
        &__php_ort_options_default->options);
    if (status) {
        return FAILURE;
    }

    status = api->SetSessionGraphOptimizationLevel(
        __php_ort_options_default->options,
            ORT_ENABLE_ALL);
    if (status) {
        return FAILURE;
    }

    status = api->SetSessionExecutionMode(
        __php_ort_options_default->options,
            ORT_PARALLEL);
    if (status) {
        return FAILURE;
    }

    /* LCOV_EXCL_STOP */
    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_OPTIONS)
{
    if (__php_ort_options_default) {
        php_ort_options_free(
            __php_ort_options_default);
    }

    return SUCCESS;
}