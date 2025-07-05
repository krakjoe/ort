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
#include "tensor.h"
#include "status.h"
#include "runtime.h"
#include "options.h"

zend_class_entry* php_ort_runtime_ce;
zend_object_handlers php_ort_runtime_handlers;

static zend_object* php_ort_runtime_create(zend_class_entry *type) {
    php_ort_runtime_t* ort = ecalloc(1, 
        sizeof(php_ort_runtime_t) + zend_object_properties_size(type));

    zend_object_std_init(
        &ort->std, php_ort_runtime_ce);

    ort->std.handlers = &php_ort_runtime_handlers;

    return &ort->std;
}

static void php_ort_runtime_destroy(zend_object* zo) {
    php_ort_runtime_t* ort =
        php_ort_runtime_fetch(zo);

    if (ort->session) {
        api->ReleaseSession(ort->session);
    }

    if (ort->model) {
        php_ort_model_release(ort->model);
    }

    if (ort->options) {
        php_ort_options_release(ort->options);
    }

    zend_object_std_dtor(zo);
}

ZEND_BEGIN_ARG_INFO_EX(php_ort_runtime_construct_arginfo, 0, 0, 1)
    ZEND_ARG_OBJ_INFO(0, model,   \\ONNX\\Model,   0)
    ZEND_ARG_OBJ_INFO(0, options, \\ONNX\\Options, 1)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Runtime, __construct)
{
    php_ort_runtime_t* ort = php_ort_runtime_fetch(Z_OBJ(EX(This)));
    zend_object *zm, *zo = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_OBJ_OF_CLASS_EX(zm, php_ort_model_ce, 1, 0)
        Z_PARAM_OPTIONAL
        Z_PARAM_OBJ_OF_CLASS_EX(zo, php_ort_options_ce, 0, 0)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_model_t* model =
        php_ort_model_fetch(zm);

    ort->model =
        model->object;

    php_ort_atomic_addref(
        &ort->model->refcount);

    if (!zo) {
        ort->options = php_ort_options_default();

        php_ort_atomic_addref(
            &ort->options->refcount);

        goto __php_ort_runtime_construct;
    }

    php_ort_options_t* options =
        php_ort_options_fetch(zo);

    ort->options =
        options->object;

    php_ort_atomic_addref(
        &ort->options->refcount);

__php_ort_runtime_construct:
    {
        php_ort_string_temp_t pass = php_ort_string_pass(ZSTR_VAL(ort->model->file));

        OrtStatus* status =
            api->CreateSession(
                php_ort_environment(),
                pass,
                ort->options->options,
                &ort->session);

        php_ort_status_flow(
            (status),
            {
                php_ort_string_free(pass);  
                return;
            },
            php_ort_status_error_ce,
            "could not start a runtime session for %s: %s",
            ZSTR_VAL(ort->model->file),
            api->GetErrorMessage(status));

        php_ort_string_free(pass);
    }
}

ZEND_BEGIN_ARG_INFO_EX(php_ort_runtime_run_arginfo, 0, 0, 1)
     ZEND_ARG_TYPE_INFO(0, input, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Runtime, run)
{
    php_ort_runtime_t* ort = php_ort_runtime_fetch(Z_OBJ(EX(This)));
    zval *input;

    ZEND_PARSE_PARAMETERS_START(1, 1);
        Z_PARAM_ARRAY(input)
    ZEND_PARSE_PARAMETERS_END();

    size_t length = zend_hash_num_elements(Z_ARRVAL_P(input));
    char** names = pecalloc(length, sizeof(char*), 0);
    OrtValue** values = pecalloc(length, sizeof(OrtValue*), 0);

    uint32_t idx = 0;
    zend_string* name;
    zval* next;

    ZEND_HASH_FOREACH_STR_KEY_VAL(Z_ARRVAL_P(input), name, next){
        php_ort_status_flow(
            ((!name) || (
                Z_TYPE_P(next) != IS_OBJECT || 
                !instanceof_function(Z_OBJCE_P(next), php_ort_tensor_interface_ce))),
            {
                pefree(names, 0);
                pefree(values, 0);
                return;
            },
            php_ort_status_model_invalidinput_ce,
            "input is required to be an associative array of name => Tensor");

        names[idx] = (char*) ZSTR_VAL(name);
        values[idx] = php_ort_tensor_value(
            php_ort_tensor_fetch(Z_OBJ_P(next)));

        idx++;
    } ZEND_HASH_FOREACH_END();

    OrtStatus* status;
    OrtValue** outputs = pecalloc(
        ort->model->counters.output, sizeof(OrtValue*), 0);

    php_ort_status_flow(
        (status = api->Run(
            ort->session, NULL,
            (const char* const*)
                names, 
            (const OrtValue* const*)
                values, 
            idx,
            (const char* const*)
                ort->model->names.output,
            ort->model->counters.output,
            outputs)),
        {
            for (idx = 0; idx <
                    zend_hash_num_elements(Z_ARRVAL_P(input)); idx++) {
                api->ReleaseValue(values[idx]);
            }

            pefree(outputs, 0);
            pefree(values, 0);
            pefree(names, 0);

            api->ReleaseStatus(status);
            return;
        },
        php_ort_status_model_runtimeexception_ce,
        "failed to run inference on %s from %s: %s",
            ZSTR_VAL(ort->model->name),
            ZSTR_VAL(ort->model->file),
            api->GetErrorMessage(status));

    for (idx = 0; idx < 
            zend_hash_num_elements(Z_ARRVAL_P(input)); idx++) {
        api->ReleaseValue(values[idx]);
    }

    array_init_size(return_value, ort->model->counters.output);
    for (idx = 0; idx < ort->model->counters.output; idx++) {
        zval zv;

        object_init_ex(&zv, php_ort_tensor_transient_ce);

        add_assoc_zval(return_value,
            ort->model->names.output[idx], &zv);

        php_ort_tensor_t *value = 
            php_ort_tensor_fetch(Z_OBJ(zv));
        value->object = 
            php_ort_tensor_object(outputs[idx]);
    }

    pefree(outputs, 0);
    pefree(values, 0);
    pefree(names, 0);
}

zend_function_entry php_ort_runtime_methods[] = {
    PHP_ME(ONNX_Runtime, __construct,       php_ort_runtime_construct_arginfo, ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Runtime, run,               php_ort_runtime_run_arginfo,       ZEND_ACC_PUBLIC)
    PHP_FE_END
};

PHP_MINIT_FUNCTION(ORT_RUNTIME)
{
    zend_class_entry ce;

    INIT_NS_CLASS_ENTRY(ce, "ONNX", "Runtime", php_ort_runtime_methods);

    php_ort_runtime_ce = zend_register_internal_class(&ce);
    php_ort_runtime_ce->create_object = php_ort_runtime_create;

    memcpy(&php_ort_runtime_handlers,
        zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    php_ort_runtime_handlers.offset    = XtOffsetOf(php_ort_runtime_t, std);
    php_ort_runtime_handlers.free_obj  = php_ort_runtime_destroy;
    php_ort_runtime_handlers.clone_obj = NULL;

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_RUNTIME)
{
    return SUCCESS;
}

PHP_RINIT_FUNCTION(ORT_RUNTIME)
{
    return SUCCESS;
}

PHP_RSHUTDOWN_FUNCTION(ORT_RUNTIME)
{
    return SUCCESS;
}