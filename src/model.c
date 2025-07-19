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
#include "status.h"
#include "model.h"
#include "iterator.h"
#include "tensor.h"

#ifdef ZTS
static MUTEX_T php_ort_model_mutex;
#endif

static HashTable php_ort_models;

zend_class_entry *php_ort_model_ce;
zend_object_handlers php_ort_model_handlers;

static void php_ort_model_free(ort_model_t *model) {
    OrtStatus* status;
    size_t idx;

    if (model->metadata) {
        api->ReleaseModelMetadata(model->metadata);
    }

    if (model->session) {
        api->ReleaseSession(model->session);
    }

    zend_string_free(model->file);
    zend_string_free(model->name);

    if (model->names.input) {
        for (idx = 0; idx < model->counters.input; idx++) {
            php_ort_status_flow(
                (status = api->AllocatorFree(
                    model->allocator, model->names.input[idx])),
                {
                    api->ReleaseStatus(status);
                    return;
                },
                php_ort_status_model_invalidmemory_ce,
                "failed to release input name at %zu for Model, something is terribly wrong",
                idx
            );
        }
        pefree(model->names.input, 1);
    }

    if (model->names.output) {
        for (idx = 0; idx < model->counters.output; idx++) {
            php_ort_status_flow(
                (status = api->AllocatorFree(
                    model->allocator, model->names.output[idx])),
                {
                    api->ReleaseStatus(status);
                    return;
                },
                php_ort_status_model_invalidmemory_ce,
                "failed to release output name at %zu for Model, something is terribly wrong",
                idx
            );
        }
        pefree(model->names.output, 1);
    }

    pefree(model, 1);
}

void php_ort_model_release(ort_model_t *model) {
    if (!model) {
        return;
    }

    if (php_ort_atomic_delref(&model->refcount) == 0){
        php_ort_model_free(model);
    }
}

static void php_ort_model_del(zval *zv) {
    php_ort_model_release(
        ((ort_model_t*) Z_PTR_P(zv)));
}

static zend_bool php_ort_model_construct(ort_model_t *model,  zend_string *name, zend_string *file){
    OrtSessionOptions* options = NULL;
    OrtStatus* status;
    size_t idx;

    model->name = php_ort_string_copy(name);
    model->file = php_ort_string_copy(file);

    php_ort_status_flow(
        (status = api->CreateSessionOptions(&options)),
        {
            api->ReleaseStatus(status);

            return 0;
        },
        php_ort_status_model_invalidoptions_ce,
        "failed to allocate SessionOptions* for Model: %s",
        api->GetErrorMessage(status));

    php_ort_status_flow(
        (status = api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL)),
        {
            api->ReleaseStatus(status);

            return 0;
        },
        php_ort_status_model_invalidoptions_ce,
        "failed to set graph optimization level in SessionOptions* for Model: %s",
        api->GetErrorMessage(status));

    php_ort_string_temp_t pass = php_ort_string_pass(ZSTR_VAL(file));

    php_ort_status_flow(
        (status = api->CreateSession(php_ort_environment(),
            pass, options, &model->session)),
        {
            api->ReleaseStatus(status);
            php_ort_string_free(pass);
            return 0;
        },
        php_ort_status_model_invalidmemory_ce,
        "failed to allocate OrtSession* for Model: %s",
        api->GetErrorMessage(status));

    php_ort_string_free(pass);

    api->ReleaseSessionOptions(options);

    php_ort_status_flow(
        (status = api->GetAllocatorWithDefaultOptions(&model->allocator)),
        {
            api->ReleaseStatus(status);

            return 0;
        },
        php_ort_status_model_invalidmemory_ce,
        "failed to allocate Allocator* for Model: %s",
        api->GetErrorMessage(status));

    php_ort_status_flow(
        (status = api->SessionGetModelMetadata(
            model->session,
            &model->metadata)),
        {
            api->ReleaseStatus(status);

            return 0;
        },
        php_ort_status_model_invalidmodel_ce,
        "failed to get metadata for Model: %s",
        api->GetErrorMessage(status));

    php_ort_status_flow(
        (status = api->SessionGetInputCount(
            model->session,
            &model->counters.input)),
        {
            api->ReleaseStatus(status);

            return 0;
        },
        php_ort_status_model_invalidmodel_ce,
        "failed to get input counter for Model: %s",
        api->GetErrorMessage(status));

    model->names.input = pecalloc(model->counters.input, sizeof(char*), 1);

    for (idx = 0; idx < model->counters.input; idx++) {
        php_ort_status_flow(
            (status = api->SessionGetInputName(
                model->session, idx, 
                model->allocator,
                &model->names.input[idx])),
            {
                api->ReleaseStatus(status);

                return 0;
            },
            php_ort_status_model_invalidinput_ce,
            "failed to get input name at %d for Model: %s",
            idx,
            api->GetErrorMessage(status));
    }

    php_ort_status_flow(
        (status = api->SessionGetOutputCount(
            model->session,
            &model->counters.output)),
        {
            api->ReleaseStatus(status);

            return 0;
        },
        php_ort_status_model_invalidmodel_ce,
        "failed to get output counter for Model: %s",
        api->GetErrorMessage(status));

    model->names.output = pecalloc(model->counters.output, sizeof(char*), 1);

    for (idx = 0; idx < model->counters.output; idx++) {
        php_ort_status_flow(
            (status = api->SessionGetOutputName(
                model->session, idx,
                model->allocator,
                &model->names.output[idx])),
            {
                api->ReleaseStatus(status);

                return 0;
            },
            php_ort_status_model_invalidoutput_ce,
            "failed to get output name at %d for Model: %s",
            idx,
            api->GetErrorMessage(status));
    }

    return 1;
}

ZEND_BEGIN_ARG_INFO_EX(php_ort_model_construct_arginfo, 0, 0, 1)
     ZEND_ARG_TYPE_INFO(0, name,  IS_STRING, 0)
     ZEND_ARG_TYPE_INFO(0, file,  IS_STRING, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model, __construct)
{
    php_ort_model_t *ort = php_ort_model_fetch(Z_OBJ(EX(This)));

    zend_string *name;
    zend_string *file = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 2);
        Z_PARAM_STR(name)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR(file)
    ZEND_PARSE_PARAMETERS_END();

#ifdef ZTS
        php_ort_status_flow(
            tsrm_mutex_lock(php_ort_model_mutex) != SUCCESS,
            return,
            php_ort_status_safetyerror_ce,
            "it was not possible to acquire the model mutex, something is terribly wrong");
#endif

    if (!(ort->object = zend_hash_find_ptr(&php_ort_models, name))) {
        char path[4096];

        php_ort_status_flow(
            !file,
            return,
            php_ort_status_model_invalidfile_ce,
            "model %s could not be loaded and no file provided",
            ZSTR_VAL(name));

        php_ort_status_flow(
            (ZSTR_LEN(file) > 4096),
            return,
            php_ort_status_model_invalidfile_ce,
            "model %s could not be created: file name too long",
            ZSTR_VAL(name));

        php_ort_status_flow(
            (NULL == VCWD_REALPATH(ZSTR_VAL(file), path)),
            return,
            php_ort_status_model_invalidfile_ce,
            "model %s could not be found, %s is not a valid path",
            ZSTR_VAL(name), ZSTR_VAL(file));

        ort_model_t *model = pecalloc(1, sizeof(ort_model_t), 1);

        model->refcount = 1;

        if (!php_ort_model_construct(model, name, file)) {
            php_ort_model_free(model);
#ifdef ZTS
            php_ort_status_flow(
                tsrm_mutex_unlock(php_ort_model_mutex) != SUCCESS,
                return,
                php_ort_status_safetyerror_ce,
                "it was not possible to release the model mutex, something is terribly wrong");
#endif
            return;
        }

        ort->object = zend_hash_add_ptr(
            &php_ort_models,
            model->name,
            model);
        php_ort_atomic_addref(&ort->object->refcount);
    } else {

#ifdef ZTS
        php_ort_status_flow(
            file && !zend_string_equals(ort->object->file, file),
            {
                php_ort_status_flow(
                    tsrm_mutex_unlock(php_ort_model_mutex) != SUCCESS,
                    return,
                    php_ort_status_safetyerror_ce,
                    "it was not possible to release the model mutex, something is terribly wrong");
                ort->object = NULL;
                return;
            },
            php_ort_status_model_invalidfile_ce,
            "a model with the name %s is already loaded from %s, the file %s does not match",
            ZSTR_VAL(ort->object->name), ZSTR_VAL(ort->object->file), ZSTR_VAL(file));
#else
        php_ort_status_flow(
            file && !zend_string_equals(ort->object->file, file),
            {
                ort->object = NULL;
                return;
            },
            php_ort_status_model_invalidfile_ce,
            "a model with the name %s is already loaded from %s, the file %s does not match",
            ZSTR_VAL(ort->object->name), ZSTR_VAL(ort->object->file), ZSTR_VAL(file));
#endif
        php_ort_atomic_addref(&ort->object->refcount);
    }

#ifdef ZTS
    php_ort_status_flow(
        tsrm_mutex_unlock(php_ort_model_mutex) != SUCCESS,
        return,
        php_ort_status_safetyerror_ce,
        "it was not possible to release the model mutex, something is terribly wrong");
#endif
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_getName_arginfo, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model, getName)
{
    php_ort_model_t *ort =
        php_ort_model_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    RETURN_STR_COPY(ort->object->name);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_getFile_arginfo, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model, getFile)
{
    php_ort_model_t *ort =
        php_ort_model_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    RETURN_STR_COPY(ort->object->file);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_getInput_arginfo, 0, 1, IS_STRING, 0)
     ZEND_ARG_TYPE_INFO(0, index, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model, getInput)
{
    php_ort_model_t *ort =
        php_ort_model_fetch(Z_OBJ(EX(This)));
    zend_long index;

    ZEND_PARSE_PARAMETERS_START(1, 1);
        Z_PARAM_LONG(index)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_status_flow(
        (index < 0 || index >= ort->object->counters.input),
        return,
        php_ort_status_model_invalidindex_ce,
        "index %zd is out of range for input",
        index);

    RETURN_STRING(ort->object->names.input[index]);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_getInputCount_arginfo, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model, getInputCount)
{
    php_ort_model_t *ort =
        php_ort_model_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    RETURN_LONG(ort->object->counters.input);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_getOutput_arginfo, 0, 1, IS_STRING, 0)
     ZEND_ARG_TYPE_INFO(0, index, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model, getOutput)
{
    php_ort_model_t *ort =
        php_ort_model_fetch(Z_OBJ(EX(This)));
    zend_long index;

    ZEND_PARSE_PARAMETERS_START(1, 1);
        Z_PARAM_LONG(index)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_status_flow(
        (index < 0 || index >= ort->object->counters.output),
        return,
        php_ort_status_model_invalidindex_ce,
        "index %zd is out of range for output",
        index);

    RETURN_STRING(ort->object->names.output[index]);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_getOutputCount_arginfo, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model, getOutputCount)
{
    php_ort_model_t *ort =
        php_ort_model_fetch(Z_OBJ(EX(This)));

    ZEND_PARSE_PARAMETERS_NONE();

    RETURN_LONG(ort->object->counters.output);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_unload_arginfo, 0, 1, _IS_BOOL, 0)
    ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model, unload)
{
    zend_string *name;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(name)
    ZEND_PARSE_PARAMETERS_END();

#ifdef ZTS
    php_ort_status_flow(
        tsrm_mutex_lock(php_ort_model_mutex) != SUCCESS,
        return,
    php_ort_status_safetyerror_ce,
    "it was not possible to acquire the model mutex, something is terribly wrong");
#endif

    RETVAL_BOOL(zend_hash_del(&php_ort_models, name) == SUCCESS);

#ifdef ZTS
    php_ort_status_flow(
        tsrm_mutex_unlock(php_ort_model_mutex) != SUCCESS,
        return,
    php_ort_status_safetyerror_ce,
    "it was not possible to release the model mutex, something is terribly wrong");
#endif
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_model_getInputIterator_arginfo, 0, 0, ORT\\Model\\Iterator, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model, getInputIterator)
{
    php_ort_model_t *ort =
        php_ort_model_fetch(Z_OBJ(EX(This)));
    ZEND_PARSE_PARAMETERS_NONE();

    php_ort_iterator_model(ort->object,
        (const char * const*) ort->object->names.input,
            ort->object->counters.input, return_value);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_model_getOutputIterator_arginfo, 0, 0, ORT\\Model\\Iterator, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model, getOutputIterator)
{
    php_ort_model_t *ort =
        php_ort_model_fetch(Z_OBJ(EX(This)));
    ZEND_PARSE_PARAMETERS_NONE();

    php_ort_iterator_model(ort->object,
        (const char * const*) ort->object->names.output,
            ort->object->counters.output, return_value);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_model_getMeta_arginfo, 0, 1, IS_MIXED, 0)
    ZEND_ARG_TYPE_INFO(0, property, IS_STRING, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Model, getMeta)
{
    php_ort_model_t* ort = php_ort_model_fetch(Z_OBJ(EX(This)));
    zend_string *property = NULL;
    char *str_value = NULL;
    int64_t int_value;
    OrtStatus* status;

    ZEND_PARSE_PARAMETERS_START(1, 1);
        Z_PARAM_STR(property)
    ZEND_PARSE_PARAMETERS_END();

    const char *prop = ZSTR_VAL(property);
    
    if (strcmp(prop, "version") == 0) {
        php_ort_status_flow(
            (status = api->ModelMetadataGetVersion(ort->object->metadata, &int_value)),
            {
                api->ReleaseStatus(status);
                return;
            },
            php_ort_status_model_invalidmodel_ce,
            "failed to get version for Model: %s",
            api->GetErrorMessage(status));
        RETURN_LONG(int_value);
    } else if (strcmp(prop, "producer") == 0) {
        php_ort_status_flow(
            (status = api->ModelMetadataGetProducerName(ort->object->metadata, ort->object->allocator, &str_value)),
            {
                api->ReleaseStatus(status);
                return;
            },
            php_ort_status_model_invalidmodel_ce,
            "failed to get producer name for Model: %s",
            api->GetErrorMessage(status));
        if (str_value) {
            RETVAL_STRING(str_value);
            php_ort_status_flow(
                (status = api->AllocatorFree(ort->object->allocator, str_value)),
                {
                    api->ReleaseStatus(status);
                    return;
                },
                php_ort_status_model_invalidmemory_ce,
                "failed to release producer name memory for Model: %s",
                api->GetErrorMessage(status));
            return;
        }
    } else if (strcmp(prop, "description") == 0) {
        php_ort_status_flow(
            (status = api->ModelMetadataGetDescription(ort->object->metadata, ort->object->allocator, &str_value)),
            {
                api->ReleaseStatus(status);
                return;
            },
            php_ort_status_model_invalidmodel_ce,
            "failed to get description for Model: %s",
            api->GetErrorMessage(status));
        if (str_value) {
            RETVAL_STRING(str_value);
            php_ort_status_flow(
                (status = api->AllocatorFree(ort->object->allocator, str_value)),
                {
                    api->ReleaseStatus(status);
                    return;
                },
                php_ort_status_model_invalidmemory_ce,
                "failed to release description memory for Model: %s",
                api->GetErrorMessage(status));
            return;
        }
    } else if (strcmp(prop, "domain") == 0) {
        php_ort_status_flow(
            (status = api->ModelMetadataGetDomain(ort->object->metadata, ort->object->allocator, &str_value)),
            {
                api->ReleaseStatus(status);
                return;
            },
            php_ort_status_model_invalidmodel_ce,
            "failed to get domain for Model: %s",
            api->GetErrorMessage(status));
        if (str_value) {
            RETVAL_STRING(str_value);
            php_ort_status_flow(
                (status = api->AllocatorFree(ort->object->allocator, str_value)),
                {
                    api->ReleaseStatus(status);
                    return;
                },
                php_ort_status_model_invalidmemory_ce,
                "failed to release domain memory for Model: %s",
                api->GetErrorMessage(status));
            return;
        }
    } else if (strcmp(prop, "graph_name") == 0) {
        php_ort_status_flow(
            (status = api->ModelMetadataGetGraphName(ort->object->metadata, ort->object->allocator, &str_value)),
            {
                api->ReleaseStatus(status);
                return;
            },
            php_ort_status_model_invalidmodel_ce,
            "failed to get graph name for Model: %s",
            api->GetErrorMessage(status));
        if (str_value) {
            RETVAL_STRING(str_value);
            php_ort_status_flow(
                (status = api->AllocatorFree(ort->object->allocator, str_value)),
                {
                    api->ReleaseStatus(status);
                    return;
                },
                php_ort_status_model_invalidmemory_ce,
                "failed to release graph name memory for Model: %s",
                api->GetErrorMessage(status));
            return;
        }
    } else if (strcmp(prop, "graph_description") == 0) {
        php_ort_status_flow(
            (status = api->ModelMetadataGetGraphDescription(ort->object->metadata, ort->object->allocator, &str_value)),
            {
                api->ReleaseStatus(status);
                return;
            },
            php_ort_status_model_invalidmodel_ce,
            "failed to get graph description for Model: %s",
            api->GetErrorMessage(status));
        if (str_value) {
            RETVAL_STRING(str_value);
            php_ort_status_flow(
                (status = api->AllocatorFree(ort->object->allocator, str_value)),
                {
                    api->ReleaseStatus(status);
                    return;
                },
                php_ort_status_model_invalidmemory_ce,
                "failed to release graph description memory for Model: %s",
                api->GetErrorMessage(status));
            return;
        }
    } else {
        /* Try to get custom metadata */
        php_ort_status_flow(
            (status = api->ModelMetadataLookupCustomMetadataMap(ort->object->metadata, ort->object->allocator, prop, &str_value)),
            {
                api->ReleaseStatus(status);
                return;
            },
            php_ort_status_model_invalidmodel_ce,
            "failed to get custom metadata for Model: %s",
            api->GetErrorMessage(status));
        if (str_value) {
            RETVAL_STRING(str_value);
            php_ort_status_flow(
                (status = api->AllocatorFree(ort->object->allocator, str_value)),
                {
                    api->ReleaseStatus(status);
                    return;
                },
                php_ort_status_model_invalidmemory_ce,
                "failed to release custom metadata memory for Model: %s",
                api->GetErrorMessage(status));
            return;
        }
    }
    
    RETURN_NULL();
}

zend_function_entry php_ort_model_methods[] = {
    PHP_ME(ONNX_Model, __construct,       php_ort_model_construct_arginfo,             ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model, getName,           php_ort_model_getName_arginfo,               ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model, getFile,           php_ort_model_getFile_arginfo,               ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model, getInput,          php_ort_model_getInput_arginfo,              ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model, getInputCount,     php_ort_model_getInputCount_arginfo,         ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model, getInputIterator,  php_ort_model_getInputIterator_arginfo,      ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model, getOutput,         php_ort_model_getOutput_arginfo,             ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model, getOutputCount,    php_ort_model_getOutputCount_arginfo,        ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model, getOutputIterator, php_ort_model_getOutputIterator_arginfo,     ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model, getMeta,           php_ort_model_getMeta_arginfo,               ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Model, unload,            php_ort_model_unload_arginfo,                ZEND_ACC_PUBLIC|ZEND_ACC_STATIC)
    PHP_FE_END
};

zend_object* php_ort_model_create(zend_class_entry *type) {
  php_ort_model_t *ort = ecalloc(1,
     sizeof(php_ort_model_t) + zend_object_properties_size(type));

   zend_object_std_init(&ort->std, type);

   ort->std.handlers = &php_ort_model_handlers;
   ort->object       = NULL;

   return &ort->std;
}

static HashTable* php_ort_model_debug(zend_object *zo, int *temp) {
    php_ort_model_t *ort = php_ort_model_fetch(zo);
    HashTable *debug;

    ALLOC_HASHTABLE(debug);
    zend_hash_init(debug, 3, NULL, ZVAL_PTR_DTOR, 0);

    if (!ort->object) {
      goto __php_ort_model_debug_return;
    }

    if (ort->object->name) {
      zval name;
  
      ZVAL_STR_COPY(&name, ort->object->name);
      zend_hash_add(debug,
        ZSTR_KNOWN(ZEND_STR_NAME), &name);
    }

    if (ort->object->file) {
      zval file;
  
      ZVAL_STR_COPY(&file, ort->object->file);
      zend_hash_add(debug,
        ZSTR_KNOWN(ZEND_STR_FILE), &file);
    }

    size_t idx;
    zval   arr;

    array_init(&arr);
    zend_hash_str_add(debug, "inputs", sizeof("inputs")-1, &arr);
    for (idx = 0; idx < ort->object->counters.input; idx++) {
      add_next_index_string(&arr, ort->object->names.input[idx]);
    }

    array_init(&arr);
    zend_hash_str_add(debug, "outputs", sizeof("outputs")-1, &arr);
    for (idx = 0; idx < ort->object->counters.output; idx++) {
      add_next_index_string(&arr, ort->object->names.output[idx]);
    }

__php_ort_model_debug_return:
    *temp = 1;

    return debug;
}

void php_ort_model_destroy(zend_object *o) {
    php_ort_model_t *ort =
        php_ort_model_fetch(o);

    php_ort_model_release(ort->object);

    zend_object_std_dtor(o);
}

PHP_MINIT_FUNCTION(ORT_MODEL)
{
    zend_class_entry ce;

#ifdef ZTS
    php_ort_model_mutex = tsrm_mutex_alloc();
#endif

    zend_hash_init(&php_ort_models, 16, NULL, php_ort_model_del, 1);

    memcpy(&php_ort_model_handlers,
        zend_get_std_object_handlers(), sizeof(zend_object_handlers));

    php_ort_model_handlers.offset = XtOffsetOf(php_ort_model_t, std);
    php_ort_model_handlers.get_debug_info = php_ort_model_debug;
    php_ort_model_handlers.free_obj = php_ort_model_destroy;
    php_ort_model_handlers.clone_obj = NULL;

    INIT_NS_CLASS_ENTRY(ce, "ORT", "Model", php_ort_model_methods);

    php_ort_model_ce = zend_register_internal_class(&ce);
    php_ort_model_ce->create_object = php_ort_model_create;

#ifdef ZEND_ACC_NOT_SERIALIZABLE
    php_ort_model_ce->ce_flags |= ZEND_ACC_NOT_SERIALIZABLE;
  #else
    php_ort_model_ce->serialize = zend_class_serialize_deny;
    php_ort_model_ce->unserialize = zend_class_unserialize_deny;
#endif

  return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_MODEL)
{
    zend_hash_destroy(&php_ort_models);
  
#ifdef ZTS
    tsrm_mutex_free(php_ort_model_mutex);
#endif

   return SUCCESS;
}
 
PHP_RINIT_FUNCTION(ORT_MODEL)
{
   return SUCCESS;
}
 
PHP_RSHUTDOWN_FUNCTION(ORT_MODEL)
{
   return SUCCESS;
}