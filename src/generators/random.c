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


 /** 
{{{
    @file generators/random.c
    @brief This API provides random generators for Tensor constructors.
    
    The constructor is overloaded:

    We shall determine the min and max derived from type:
    Tensor\Generator Tensor\Generator\Random(int $type)

    We shall determine the max derived from type, verifying the minimum is within range:
    Tensor\Generator Tensor\Generator\Random(int $type, mixed $min)

    We shall verify the min and max are within range:
    Tensor\Generator Tensor\Generator\Random(int $type, mixed $min, mixed $max)

    For example:
        $tensor = new Tensor\Transient([24, 24],
            new Tensor\Generator\Random(Tensor::FLOAT));
        Will generate a 24x24 tensor with float values in the range 0-1 (never inclusive of 1.0)

        $tensor = new Tensor\Transient([24, 24],
            new Tensor\Generator\Random(Tensor::UINT8, 0, 255));
        Will generate a 24x24 tensor with UINT8 values in the range 0-255

        $tensor = new Tensor\Transient([24, 24],
            new Tensor\Generator\Random(Tensor::INT8));
        Will generate a 24x24 tensor with INT8 values in the range INT8_MIN to INT8_MAX

        An exception will be thrown if min exceeds max, of if any parameter is out of range.
}}} */
#include "status.h"
#include "generators.h"
#include "generators/random.h"

#if PHP_VERSION_ID >= 80200
#include <ext/random/php_random.h>
#else
#include <ext/standard/php_random.h>
#endif

zend_class_entry* php_ort_generator_random_ce;

typedef struct _php_ort_generator_random_context_t{
    zend_long type;
    zval min;
    zval max;
} php_ort_generator_random_context_t;

static zend_always_inline void
    php_ort_generator_random_invoke(
        php_ort_generator_t* generator, zval* zv) {
    php_ort_generator_random_context_t* context =
        (php_ort_generator_random_context_t*) generator->context;

    switch (context->type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
            double min = Z_DVAL(context->min);
            double max = Z_DVAL(context->max);
            double scale =
                (double) php_mt_rand() /
                    ((double) PHP_MT_RAND_MAX + 1.0);

            if (scale > 0.999999) {
                scale = 0.999999;
            }

            if (scale < 0.0) {
                scale = 0;
            }

            ZVAL_DOUBLE(zv, min + scale * (max - min));
            break;
        }
        
        default: {
            ZVAL_LONG(zv, php_mt_rand_range(
                Z_LVAL(context->min), Z_LVAL(context->max)));
            break;
        }
    }
}

static zend_object* php_ort_generator_random_create(
    zend_class_entry* ce) {
    php_ort_generator_t* generator =
        zend_object_alloc(
            sizeof(php_ort_generator_t), ce);

    zend_object_std_init(&generator->std, ce);
    object_properties_init(&generator->std, ce);

    generator->std.handlers =
        &php_ort_generator_handlers;

    generator->invoke = php_ort_generator_random_invoke;
    generator->context =
        ecalloc(1, sizeof(php_ort_generator_random_context_t));

    return &generator->std;
}

ZEND_BEGIN_ARG_INFO_EX(
    php_ort_generator_random__construct_arginfo, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, type, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, min, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, max, IS_LONG, 0)
ZEND_END_ARG_INFO()

#define ORT_GENERATOR_RANDOM_CHECK_TYPE(type, var, name) \
do {                                                     \
    if (Z_TYPE_P(var) != IS_##type) {                    \
        zend_throw_exception_ex(                         \
            php_ort_status_error_ce, 0,                  \
            "Expected "#type" for "#name", got %s",      \
            zend_zval_value_name(var));                  \
        return;                                          \
    }                                                    \
} while(0)

#define ORT_GENERATOR_RANDOM_CHECK_SIGNED_RANGE(var, name, min, max)\
do {                                                                  \
    if (Z_LVAL_P(var) < min || Z_LVAL_P(var) > max) {                 \
        zend_throw_exception_ex(                                      \
            php_ort_status_error_ce, 0,                               \
            "Value for "#name" must be between "                      \
                ZEND_LONG_FMT" and "                                  \
                ZEND_LONG_FMT", got "ZEND_LONG_FMT"",                 \
            (zend_long) min,                                          \
            (zend_long) max,                                          \
            Z_LVAL_P(var));                                           \
        return;                                                       \
    }                                                                 \
} while(0)

#define ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_RANGE(var, name, max)     \
do {                                                                  \
    if (Z_LVAL_P(var) < 0 || Z_LVAL_P(var) > max) {                   \
        zend_throw_exception_ex(                                      \
            php_ort_status_error_ce, 0,                               \
            "Value for "#name" must be between 0 and "                \
            "%d, got "ZEND_LONG_FMT"",                                \
            max,                                                      \
            Z_LVAL_P(var));                                           \
        return;                                                       \
    }                                                                 \
} while(0)

PHP_METHOD(Generator_RandomInt, __construct) {
    php_ort_generator_t* generator =
        php_ort_generator_fetch(
            Z_OBJ_P(getThis()));
    php_ort_generator_random_context_t* context =
        (php_ort_generator_random_context_t*)
            generator->context;

    memset(context, 0, sizeof(
        php_ort_generator_random_context_t));

    zval* min = NULL;
    zval* max = NULL;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_LONG(context->type)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(min)
        Z_PARAM_ZVAL(max)
    ZEND_PARSE_PARAMETERS_END();

    if (ZEND_NUM_ARGS() == 1) {
        /* only a type provided, we set range */
        switch (context->type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                ZVAL_DOUBLE(&context->min, 0.0);
                ZVAL_DOUBLE(&context->max, 1.0);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                ZVAL_LONG(&context->min, INT8_MIN);
                ZVAL_LONG(&context->max, INT8_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                ZVAL_LONG(&context->min, INT16_MIN);
                ZVAL_LONG(&context->max, INT16_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                ZVAL_LONG(&context->min, INT32_MIN);
                ZVAL_LONG(&context->max, INT32_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                ZVAL_LONG(&context->min, INT64_MIN);
                ZVAL_LONG(&context->max, INT64_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                ZVAL_LONG(&context->min, 0);
                ZVAL_LONG(&context->max, UINT8_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                ZVAL_LONG(&context->min, 0);
                ZVAL_LONG(&context->max, UINT16_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                ZVAL_LONG(&context->min, 0);
                ZVAL_LONG(&context->max, UINT32_MAX);
                break;

            default:
                zend_throw_exception_ex(
                    php_ort_status_error_ce, 0,
                    "Unsupported type for Random generator: %s",
                    php_ort_type_name(context->type));
                return;
        }
    } else if (ZEND_NUM_ARGS() == 2) {
        /* type and min provided */

#define ORT_GENERATOR_RANDOM_CHECK_SIGNED_CASE(          \
    type, minimum, maximum) do {                         \
    ORT_GENERATOR_RANDOM_CHECK_TYPE(type, min, "min");   \
    ORT_GENERATOR_RANDOM_CHECK_SIGNED_RANGE(             \
        min, "min", minimum, maximum);                   \
    ZVAL_COPY(&context->min, min);                       \
    ZVAL_LONG(&context->max, maximum);                   \
} while(0)

#define ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_CASE(        \
    type, maximum) do {                                  \
    ORT_GENERATOR_RANDOM_CHECK_TYPE(type, min, "min");   \
    ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_RANGE(           \
        min, "min", maximum);                            \
    ZVAL_COPY(&context->min, min);                       \
    ZVAL_LONG(&context->max, maximum);                   \
} while(0)

        switch (context->type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                ORT_GENERATOR_RANDOM_CHECK_TYPE(
                    DOUBLE, min, "min");
                ZVAL_COPY(&context->min, min);
                ZVAL_DOUBLE(&context->max, 1.0);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                ORT_GENERATOR_RANDOM_CHECK_SIGNED_CASE(LONG, INT8_MIN, INT8_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                ORT_GENERATOR_RANDOM_CHECK_SIGNED_CASE(LONG, INT16_MIN, INT16_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                ORT_GENERATOR_RANDOM_CHECK_SIGNED_CASE(LONG, INT32_MIN, INT32_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                ORT_GENERATOR_RANDOM_CHECK_SIGNED_CASE(LONG, INT64_MIN, INT64_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_CASE(LONG, UINT8_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_CASE(LONG, UINT16_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_CASE(LONG, UINT32_MAX);
                break;

            default:
                zend_throw_exception_ex(
                    php_ort_status_error_ce, 0,
                    "Unsupported type for Random generator: %s",
                    php_ort_type_name(context->type));
                return;
        }

#undef ORT_GENERATOR_RANDOM_CHECK_SIGNED_CASE
#undef ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_CASE
    } else if (ZEND_NUM_ARGS() == 3) {
        /* type, min and max provided */

#define ORT_GENERATOR_RANDOM_CHECK_SIGNED_CASE(        \
    type, minimum, maximum) do {                       \
    ORT_GENERATOR_RANDOM_CHECK_TYPE(                   \
        type, min, "min");                             \
    ORT_GENERATOR_RANDOM_CHECK_TYPE(                   \
        type, max, "max");                             \
    ORT_GENERATOR_RANDOM_CHECK_SIGNED_RANGE(           \
        min, "min", minimum, maximum);                 \
    ORT_GENERATOR_RANDOM_CHECK_SIGNED_RANGE(           \
        max, "max", minimum, maximum);                 \
    if (Z_LVAL_P(min) > Z_LVAL_P(max)) {               \
        zend_throw_exception_ex(                       \
            php_ort_status_error_ce, 0,                \
            "Min value must be less than or equal to " \
            "max value for Random generator");         \
        return;                                        \
    }                                                  \
    ZVAL_COPY(&context->min, min);                     \
    ZVAL_COPY(&context->max, max);                     \
} while(0)

#define ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_CASE(      \
    type,  maximum) do {                               \
    ORT_GENERATOR_RANDOM_CHECK_TYPE(                   \
        type, min, "min");                             \
    ORT_GENERATOR_RANDOM_CHECK_TYPE(                   \
        type, max, "max");                             \
    ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_RANGE(         \
        min, "min", maximum);                          \
    ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_RANGE(         \
        max, "max", maximum);                          \
    if (Z_LVAL_P(min) > Z_LVAL_P(max)) {               \
        zend_throw_exception_ex(                       \
            php_ort_status_error_ce, 0,                \
            "Min value must be less than or equal to " \
            "max value for Random generator");         \
        return;                                        \
    }                                                  \
    ZVAL_COPY(&context->min, min);                     \
    ZVAL_COPY(&context->max, max);                     \
} while(0)

        switch (context->type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                ORT_GENERATOR_RANDOM_CHECK_TYPE(
                    DOUBLE, min, "min");
                ORT_GENERATOR_RANDOM_CHECK_TYPE(
                    DOUBLE, max, "max");
                ZVAL_COPY(&context->min, min);
                ZVAL_COPY(&context->max, max);
            break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                ORT_GENERATOR_RANDOM_CHECK_SIGNED_CASE(LONG, INT8_MIN, INT8_MAX);
                break;
            
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                ORT_GENERATOR_RANDOM_CHECK_SIGNED_CASE(LONG, INT16_MIN, INT16_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                ORT_GENERATOR_RANDOM_CHECK_SIGNED_CASE(LONG, INT32_MIN, INT32_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                ORT_GENERATOR_RANDOM_CHECK_SIGNED_CASE(LONG, INT64_MIN, INT64_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_CASE(LONG, UINT8_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_CASE(LONG, UINT16_MAX);
                break;

            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_CASE(LONG, UINT32_MAX);
                break;

            default:
                zend_throw_exception_ex(
                    php_ort_status_error_ce, 0,
                    "Unsupported type for Random generator: %s",
                    php_ort_type_name(context->type));
                return;
        }
#undef ORT_GENERATOR_RANDOM_CHECK_SIGNED_CASE
#undef ORT_GENERATOR_RANDOM_CHECK_UNSIGNED_CASE
    }
}

#undef ORT_GENERATOR_RANDOM_CHECK_TYPE

zend_function_entry php_ort_generator_random_methods[] = {
    PHP_ME(Generator_RandomInt, __construct,
        php_ort_generator_random__construct_arginfo, ZEND_ACC_PUBLIC)
};

PHP_MINIT_FUNCTION(ORT_GENERATORS_RANDOM)
{
    zend_class_entry ce;

    INIT_NS_CLASS_ENTRY(ce,
        "ORT\\Tensor\\Generator",
        "Random", php_ort_generator_random_methods);
    php_ort_generator_random_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_generator_internal_ce);
    php_ort_generator_random_ce->create_object =
        php_ort_generator_random_create;
    php_ort_generator_random_ce->ce_flags |= ZEND_ACC_FINAL;

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_GENERATORS_RANDOM) 
{
    return SUCCESS;
}