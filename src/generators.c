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

#include "generators.h"
#include "generators/random.h"

zend_class_entry* php_ort_generator_ce;
zend_class_entry* php_ort_generator_internal_ce;
zend_object_handlers php_ort_generator_handlers;

typedef struct _php_ort_generator_context_t {
    zend_fcall_info       fci;
    zend_fcall_info_cache fcc;
} php_ort_generator_context_t;

static zend_object* php_ort_generator_create_user(
    zend_class_entry* ce) {
    php_ort_generator_t* generator = zend_object_alloc(
        sizeof(php_ort_generator_t), ce);

    zend_object_std_init(&generator->std, ce);
    object_properties_init(&generator->std, ce);

    generator->std.handlers =
        &php_ort_generator_handlers;

    php_ort_generator_context_t* context =
        emalloc(sizeof(php_ort_generator_context_t));

    memset(&context->fci, 0, sizeof(context->fci));
    memset(&context->fcc, 0, sizeof(context->fcc));

    context->fci.size = sizeof(context->fci);
    context->fci.object = &generator->std;
    context->fcc.object = &generator->std;
    context->fcc.function_handler =
        zend_hash_find_ptr(
            &ce->function_table,
            ZSTR_KNOWN(ZEND_STR_MAGIC_INVOKE));
    context->fcc.calling_scope = ce;
    context->fcc.called_scope = ce;

    generator->context = context;

    return &generator->std;
}

static zend_object* php_ort_generator_create_internal(
    zend_class_entry* ce) {
    php_ort_generator_t* generator = zend_object_alloc(
        sizeof(php_ort_generator_t), ce);

    zend_object_std_init(&generator->std, ce);
    object_properties_init(&generator->std, ce);

    generator->std.handlers =
        &php_ort_generator_handlers;

    return &generator->std;
}

static void php_ort_generator_destroy(zend_object* zo) {
    php_ort_generator_t* generator =
        php_ort_generator_fetch(zo);

    if (generator->context) {
        efree(generator->context);
    }

    zend_object_std_dtor(zo);
}

static zend_always_inline zend_bool
    php_ort_generator_invoke_user(
        php_ort_generator_t* generator,
        ONNXTensorElementDataType type,
        void* target) {
    zval rv;
    php_ort_generator_context_t* context =
        (php_ort_generator_context_t*)
            generator->context;

    ZVAL_UNDEF(&rv);

    ZVAL_OBJ(
        &context->fci.function_name,
        &generator->std);
    context->fci.retval = &rv;

    zend_call_function(
        &context->fci, &context->fcc);
    php_ort_tensor_store(
        type, target, &rv);
    zval_ptr_dtor(&rv);

    return !EG(exception);
}

static zend_always_inline zend_bool
    php_ort_generator_invoke_internal(
        php_ort_generator_t* generator,
        ONNXTensorElementDataType type,
        void* target) {
    zval rv;

    ZVAL_UNDEF(&rv);
    generator->invoke(generator, &rv);
    php_ort_tensor_store(
        type, target, &rv);
    zval_ptr_dtor(&rv);

    return !Z_ISUNDEF(rv);
}

zend_bool php_ort_generator_invoke(
    zval* zv,
    ONNXTensorElementDataType type,
    void* target) {
    zval rv;
    php_ort_generator_t* generator =
        php_ort_generator_fetch(Z_OBJ_P(zv));

    if (instanceof_function(
        generator->std.ce, php_ort_generator_internal_ce)) {
        return php_ort_generator_invoke_internal(
            generator, type, target);
    }

    if (instanceof_function(
        generator->std.ce, php_ort_generator_ce)) {
        return php_ort_generator_invoke_user(
            generator, type, target);
    }

    /* unexpected */
    return 0;
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(
    php_ort_generator_invoke_arginfo, 0, 0, IS_MIXED, 0)
ZEND_END_ARG_INFO()

zend_function_entry php_ort_generator_methods[] = {
    PHP_ABSTRACT_ME(Generator, __invoke, php_ort_generator_invoke_arginfo)
    PHP_FE_END
};

PHP_METHOD(Generator, __invoke) {
    ZEND_PARSE_PARAMETERS_NONE();
}

zend_function_entry php_ort_generator_internal_methods[] = {
    PHP_ME(Generator, __invoke,
        php_ort_generator_invoke_arginfo,
        ZEND_ACC_PUBLIC|ZEND_ACC_FINAL)
    PHP_FE_END
};

PHP_MINIT_FUNCTION(ORT_GENERATORS) 
{
    zend_class_entry ce;

    INIT_NS_CLASS_ENTRY(ce, "ORT\\Tensor",
        "Generator", php_ort_generator_methods);
    php_ort_generator_ce =
        zend_register_internal_class(&ce);
    php_ort_generator_ce->create_object =
        php_ort_generator_create_user;
    
    memcpy(
        &php_ort_generator_handlers,
        zend_get_std_object_handlers(),
        sizeof(zend_object_handlers));

    php_ort_generator_handlers.offset =
        XtOffsetOf(php_ort_generator_t, std);
    php_ort_generator_handlers.free_obj =
        php_ort_generator_destroy;
    php_ort_generator_handlers.clone_obj = NULL;

    INIT_NS_CLASS_ENTRY(ce, "ORT\\Tensor\\Generator",
        "Internal", php_ort_generator_internal_methods);
    php_ort_generator_internal_ce =
        zend_register_internal_class_ex(&ce, php_ort_generator_ce);
    php_ort_generator_internal_ce->create_object =
        php_ort_generator_create_internal;

    PHP_MINIT(ORT_GENERATORS_RANDOM)(INIT_FUNC_ARGS_PASSTHRU);

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_GENERATORS)
{
    PHP_MSHUTDOWN(ORT_GENERATORS_RANDOM)(INIT_FUNC_ARGS_PASSTHRU);

    return SUCCESS;
}