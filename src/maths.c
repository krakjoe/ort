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
#include "status.h"

#include "maths.h"
#include "maths/api.h"
#include "maths/cast.h"
#include "maths/result.h"
#include "maths/pool.h"
#include "maths/promotion.h"

typedef struct _php_ort_math_schema_t {
    const ort_math_promotion_schema_t* schema;
    zend_string*                            symbol;
    zend_object std;
} php_ort_math_schema_t;

zend_class_entry *php_ort_math_schema_ce;
zend_object_handlers php_ort_math_schema_handlers;

static zend_always_inline php_ort_math_schema_t* php_ort_math_schema_fetch(zend_object *o) {
    return (php_ort_math_schema_t*) (((char*) o) - XtOffsetOf(php_ort_math_schema_t, std));
}

static zend_object* php_ort_math_schema_create(zend_class_entry *ce) {
    php_ort_math_schema_t *ort = 
        zend_object_alloc(sizeof(php_ort_math_schema_t), ce);
        
    zend_object_std_init(&ort->std, ce);
    object_properties_init(&ort->std, ce);

    ort->std.handlers = &php_ort_math_schema_handlers;
    return &ort->std;
}

static HashTable* php_ort_math_schema_debug(zend_object *zo, int *temp) {
    php_ort_math_schema_t *ort = php_ort_math_schema_fetch(zo);
    HashTable *debug;

    ALLOC_HASHTABLE(debug);
    zend_hash_init(debug, 3, NULL, ZVAL_PTR_DTOR, 0);

    if (ort->symbol) {
        zval symbol;

        ZVAL_STR_COPY(&symbol, ort->symbol);

        zend_hash_str_update(debug,
            "symbol", sizeof("symbol") - 1,
            &symbol);
    }

    *temp = 1;

    return debug;
}

void php_ort_math_schema_free(zend_object *zo) {
    php_ort_math_schema_t* ort = php_ort_math_schema_fetch(zo);

    if (ort->symbol) {
        zend_string_release(ort->symbol);
    }

    zend_object_std_dtor(zo);
}

#define ORT_MATH_UNARY_FUNCTION_IMPL(fname)                    \
    ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(                    \
        php_ort_math_##fname##_arginfo, 0, 1, ORT\\Tensor, 0)  \
        ZEND_ARG_OBJ_INFO(0, tensor, ORT\\Tensor, 0)           \
    ZEND_END_ARG_INFO()                                        \
    PHP_NAMED_FUNCTION(php_ort_math_##fname)                   \
    {                                                          \
        zval *tensor_zv;                                       \
        ZEND_PARSE_PARAMETERS_START(1, 1)                      \
            Z_PARAM_OBJECT_OF_CLASS(tensor_zv,                 \
                php_ort_tensor_interface_ce)                   \
        ZEND_PARSE_PARAMETERS_END();                           \
        php_ort_tensor_t* tensor_ort =                         \
            php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));          \
        ort_tensor_t* result =                                 \
            ort_math_result_##fname(tensor_ort->object);       \
        object_init_ex(return_value,                           \
            php_ort_tensor_transient_ce);                      \
        php_ort_tensor_t* rv =                                 \
            php_ort_tensor_fetch(Z_OBJ_P(return_value));       \
        rv->object = result;                                   \
    }

#define ORT_MATH_BINARY_FUNCTION_IMPL(fname)                    \
    ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(                     \
        php_ort_math_##fname##_arginfo, 0, 2, ORT\\Tensor, 0)   \
        ZEND_ARG_OBJ_INFO(0, tensor_a, ORT\\Tensor, 0)          \
        ZEND_ARG_INFO(0, tensor_b_or_scalar)                    \
    ZEND_END_ARG_INFO()                                         \
    PHP_NAMED_FUNCTION(php_ort_math_##fname)                    \
    {                                                           \
        zval *tensor_a_zv, *tensor_b_or_scalar;                 \
        ZEND_PARSE_PARAMETERS_START(2, 2)                       \
            Z_PARAM_OBJECT_OF_CLASS(tensor_a_zv,                \
                php_ort_tensor_interface_ce)                    \
            Z_PARAM_ZVAL(tensor_b_or_scalar)                    \
        ZEND_PARSE_PARAMETERS_END();                            \
        php_ort_tensor_t* tensor_a_ort =                        \
            php_ort_tensor_fetch(Z_OBJ_P(tensor_a_zv));         \
        ort_tensor_t* result = NULL;                            \
        if (Z_TYPE_P(tensor_b_or_scalar) == IS_OBJECT &&        \
            instanceof_function(Z_OBJCE_P(tensor_b_or_scalar),  \
                php_ort_tensor_interface_ce)) {                 \
            php_ort_tensor_t* tensor_b_ort =                    \
                php_ort_tensor_fetch(Z_OBJ_P(tensor_b_or_scalar)); \
            result = ort_math_result_##fname(                   \
                tensor_a_ort->object, tensor_b_ort->object);    \
        } else if (Z_TYPE_P(tensor_b_or_scalar) == IS_LONG ||   \
                   Z_TYPE_P(tensor_b_or_scalar) == IS_DOUBLE || \
                   (Z_TYPE_P(tensor_b_or_scalar) == IS_TRUE ||  \
                    Z_TYPE_P(tensor_b_or_scalar) == IS_FALSE)) { \
            result = ort_math_result_##fname##_scalar(          \
                tensor_a_ort->object, tensor_b_or_scalar);      \
        } else {                                                \
            php_ort_status_throw(                               \
                php_ort_status_math_invalidtype_ce,             \
                #fname ": second argument must be a Tensor or numeric value"); \
            return;                                             \
        }                                                       \
        object_init_ex(return_value,                            \
            php_ort_tensor_transient_ce);                       \
        php_ort_tensor_t* rv =                                  \
            php_ort_tensor_fetch(Z_OBJ_P(return_value));        \
        rv->object = result;                                    \
    }


#define ORT_MATH_REDUCTION_TENSOR_FUNCTION_IMPL(fname)             \
    ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(                        \
            php_ort_math_reduce_tensor_##fname##_arginfo,          \
                0, 1, ORT\\Tensor, 0)                             \
        ZEND_ARG_OBJ_INFO(0, tensor, ORT\\Tensor, 0)              \
    ZEND_END_ARG_INFO()                                            \
    PHP_NAMED_FUNCTION(php_ort_math_reduce_tensor_##fname)         \
    {                                                              \
        zval *tensor_zv;                                           \
        ZEND_PARSE_PARAMETERS_START(1, 1)                          \
            Z_PARAM_OBJECT_OF_CLASS(tensor_zv,                     \
                php_ort_tensor_interface_ce)                       \
        ZEND_PARSE_PARAMETERS_END();                               \
        php_ort_tensor_t* tensor_ort =                             \
            php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));              \
        ort_tensor_t* result =                                     \
            ort_math_result_reduce_tensor_##fname(                 \
                tensor_ort->object);                               \
        object_init_ex(return_value,                               \
            php_ort_tensor_transient_ce);                          \
        php_ort_tensor_t* rv =                                     \
            php_ort_tensor_fetch(Z_OBJ_P(return_value));           \
        rv->object = result;                                       \
    }

// Macro for reduction functions (Tensor, axis, keepdims)
#define ORT_MATH_REDUCTION_AXIS_FUNCTION_IMPL(fname)               \
    ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(                        \
            php_ort_math_reduce_axis_##fname##_arginfo, 0, 2, ORT\\Tensor, 0) \
        ZEND_ARG_OBJ_INFO(0, tensor, ORT\\Tensor, 0)              \
        ZEND_ARG_TYPE_INFO(0, axis, IS_LONG, 0)                    \
        ZEND_ARG_TYPE_INFO(0, keepdims, _IS_BOOL, 0)               \
    ZEND_END_ARG_INFO()                                            \
    PHP_NAMED_FUNCTION(php_ort_math_reduce_axis_##fname)           \
    {                                                              \
        zval *tensor_zv;                                           \
        zend_long axis = 0;                                        \
        zend_bool keepdims = 0;                                    \
        ZEND_PARSE_PARAMETERS_START(2, 3)                          \
            Z_PARAM_OBJECT_OF_CLASS(tensor_zv,                     \
                php_ort_tensor_interface_ce)                       \
            Z_PARAM_LONG(axis)                                     \
            Z_PARAM_OPTIONAL                                       \
            Z_PARAM_BOOL(keepdims)                                 \
        ZEND_PARSE_PARAMETERS_END();                               \
        php_ort_tensor_t* tensor_ort =                             \
            php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));              \
        ort_tensor_t* result = ort_math_result_reduce_axis_##fname(\
            tensor_ort->object, axis, keepdims);                   \
        object_init_ex(return_value,                               \
            php_ort_tensor_transient_ce);                          \
        php_ort_tensor_t* rv =                                     \
            php_ort_tensor_fetch(Z_OBJ_P(return_value));           \
        rv->object = result;                                       \
    }
/* Mathematical functions in ORT\Math namespace */

ORT_MATH_BINARY_FUNCTION_IMPL(add)
ORT_MATH_BINARY_FUNCTION_IMPL(multiply)
ORT_MATH_BINARY_FUNCTION_IMPL(subtract)
ORT_MATH_BINARY_FUNCTION_IMPL(divide)

ORT_MATH_BINARY_FUNCTION_IMPL(pow)
ORT_MATH_BINARY_FUNCTION_IMPL(mod)

ORT_MATH_UNARY_FUNCTION_IMPL(sin)
ORT_MATH_UNARY_FUNCTION_IMPL(cos)
ORT_MATH_UNARY_FUNCTION_IMPL(tan)
ORT_MATH_UNARY_FUNCTION_IMPL(asin)
ORT_MATH_UNARY_FUNCTION_IMPL(acos)
ORT_MATH_UNARY_FUNCTION_IMPL(arccos)
ORT_MATH_UNARY_FUNCTION_IMPL(arcsin)
ORT_MATH_UNARY_FUNCTION_IMPL(arctan)
ORT_MATH_UNARY_FUNCTION_IMPL(arccosh)
ORT_MATH_UNARY_FUNCTION_IMPL(arcsinh)
ORT_MATH_UNARY_FUNCTION_IMPL(arctanh)
ORT_MATH_UNARY_FUNCTION_IMPL(atan)
ORT_MATH_UNARY_FUNCTION_IMPL(sinh)
ORT_MATH_UNARY_FUNCTION_IMPL(cosh)
ORT_MATH_UNARY_FUNCTION_IMPL(tanh)
ORT_MATH_UNARY_FUNCTION_IMPL(exp)
ORT_MATH_UNARY_FUNCTION_IMPL(exp2)
ORT_MATH_UNARY_FUNCTION_IMPL(log)
ORT_MATH_UNARY_FUNCTION_IMPL(log2)
ORT_MATH_UNARY_FUNCTION_IMPL(log10)
ORT_MATH_UNARY_FUNCTION_IMPL(abs)
ORT_MATH_UNARY_FUNCTION_IMPL(recip)
ORT_MATH_UNARY_FUNCTION_IMPL(neg)
ORT_MATH_UNARY_FUNCTION_IMPL(cbrt)
ORT_MATH_UNARY_FUNCTION_IMPL(ceil)
ORT_MATH_UNARY_FUNCTION_IMPL(floor)
ORT_MATH_UNARY_FUNCTION_IMPL(round)
ORT_MATH_UNARY_FUNCTION_IMPL(trunc)
ORT_MATH_UNARY_FUNCTION_IMPL(sign)
ORT_MATH_UNARY_FUNCTION_IMPL(sqrt)

ORT_MATH_REDUCTION_TENSOR_FUNCTION_IMPL(min)
ORT_MATH_REDUCTION_AXIS_FUNCTION_IMPL(min)

ORT_MATH_REDUCTION_TENSOR_FUNCTION_IMPL(max)
ORT_MATH_REDUCTION_AXIS_FUNCTION_IMPL(max)

ORT_MATH_REDUCTION_TENSOR_FUNCTION_IMPL(mean)
ORT_MATH_REDUCTION_AXIS_FUNCTION_IMPL(mean)

ORT_MATH_REDUCTION_TENSOR_FUNCTION_IMPL(sum)
ORT_MATH_REDUCTION_AXIS_FUNCTION_IMPL(sum)

ORT_MATH_REDUCTION_AXIS_FUNCTION_IMPL(softmax)

ORT_MATH_REDUCTION_TENSOR_FUNCTION_IMPL(argmax)
ORT_MATH_REDUCTION_AXIS_FUNCTION_IMPL(argmax)

/* Dot reduction function */
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_dot_arginfo, 0, 1, ORT\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor_a, ORT\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor_b, ORT\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_NAMED_FUNCTION(php_ort_math_dot)
{
    zval *tensor_a, *tensor_b;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(tensor_a, php_ort_tensor_interface_ce)
        Z_PARAM_OBJECT_OF_CLASS(tensor_b, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_a_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_a));
    php_ort_tensor_t* tensor_b_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_b));
    ort_tensor_t* result = ort_math_result_dot(
        tensor_a_ort->object, tensor_b_ort->object);

    /* Create PHP tensor object for result */
    object_init_ex(return_value,
        php_ort_tensor_transient_ce);
    php_ort_tensor_t* rv =
        php_ort_tensor_fetch(Z_OBJ_P(return_value));
    rv->object = result;
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_matmul_arginfo, 0, 2, ORT\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, matrix_a, ORT\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, matrix_b, ORT\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_NAMED_FUNCTION(php_ort_math_matmul)
{
    zval *matrix_a_zv, *matrix_b_zv;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(matrix_a_zv, php_ort_tensor_interface_ce)
        Z_PARAM_OBJECT_OF_CLASS(matrix_b_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* matrix_a_ort = php_ort_tensor_fetch(Z_OBJ_P(matrix_a_zv));
    php_ort_tensor_t* matrix_b_ort = php_ort_tensor_fetch(Z_OBJ_P(matrix_b_zv));
    
    ort_tensor_t* result = ort_math_result_matmul(matrix_a_ort->object, matrix_b_ort->object);

    object_init_ex(return_value,
        php_ort_tensor_transient_ce);
    php_ort_tensor_t* rv =
        php_ort_tensor_fetch(Z_OBJ_P(return_value));
    rv->object = result;
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(php_ort_math_backend_arginfo, 0, 0, MAY_BE_ARRAY|MAY_BE_FALSE)
ZEND_END_ARG_INFO()

PHP_NAMED_FUNCTION(php_ort_math_backend)
{
    ZEND_PARSE_PARAMETERS_NONE();

#if defined(ORT_BACKEND_CPU_ENABLED) || defined(ORT_BACKEND_GPU_ENABLED)
    array_init(return_value);
# if defined(ORT_BACKEND_GPU_ENABLED)
    add_next_index_string(return_value, ORT_BACKEND_GPU_NAME);
# endif
# if defined(ORT_BACKEND_CPU_ENABLED)
    add_next_index_string(return_value, ORT_BACKEND_CPU_NAME);
# endif
#else
    RETURN_FALSE;
#endif
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(php_ort_math_backend_cpu_arginfo, 0, 0, MAY_BE_STRING|MAY_BE_FALSE)
ZEND_END_ARG_INFO()

PHP_NAMED_FUNCTION(php_ort_math_backend_cpu)
{
    ZEND_PARSE_PARAMETERS_NONE();

#if defined(ORT_BACKEND_CPU_ENABLED)
    RETURN_STRING(ORT_BACKEND_CPU_NAME);
#else
    RETURN_FALSE;
#endif
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(php_ort_math_backend_gpu_arginfo, 0, 0, MAY_BE_STRING|MAY_BE_FALSE)
ZEND_END_ARG_INFO()

PHP_NAMED_FUNCTION(php_ort_math_backend_gpu)
{
    ZEND_PARSE_PARAMETERS_NONE();

#if defined(ORT_BACKEND_GPU_ENABLED)
    RETURN_STRING(ORT_BACKEND_GPU_NAME);
#else
    RETURN_FALSE;
#endif
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_math_cores_arginfo, 0, 0, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, max, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

PHP_NAMED_FUNCTION(php_ort_math_cores)
{
    zend_bool max = false;

    ZEND_PARSE_PARAMETERS_START(0, 1);
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(max)
    ZEND_PARSE_PARAMETERS_END();

    if (max) {
        RETURN_LONG(
            ort_pool_max());
    }

    RETURN_LONG(
        ort_pool_cores());
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_math_threshold_arginfo, 0, 0, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, default, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

PHP_NAMED_FUNCTION(php_ort_math_threshold)
{
    zend_bool _default = false;

    ZEND_PARSE_PARAMETERS_START(0, 1);
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(_default)
    ZEND_PARSE_PARAMETERS_END();

    if (_default) {
        RETURN_LONG(ORT_SCALE_THRESHOLD);
    }

    RETURN_LONG(
        ort_pool_threshold());
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_math_scale_arginfo, 0, 2, IS_MIXED, 0)
    ZEND_ARG_TYPE_INFO(0, cores, IS_LONG, 0)
    ZEND_ARG_CALLABLE_INFO(0, code, 0)
    ZEND_ARG_TYPE_INFO(0, threshold, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_NAMED_FUNCTION(php_ort_math_scale)
{
    ort_pool_scale_t scale =
        (ort_pool_scale_t) {
            .kind = ORT_POOL_SCALE_CORES
    };

    zend_fcall_info fci = empty_fcall_info;
    zend_fcall_info_cache fcc = empty_fcall_info_cache;

    ZEND_PARSE_PARAMETERS_START(2, 3);
        Z_PARAM_LONG(scale.cores)
        Z_PARAM_FUNC(fci, fcc)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(scale.threshold)
    ZEND_PARSE_PARAMETERS_END();

    if (ZEND_NUM_ARGS() > 2) {
        /* Request threshold scaling */
        scale.kind |=
            ORT_POOL_SCALE_THRESHOLD;
    }

    scale = ort_pool_scale(&scale);

    php_ort_status_flow(
        (scale.kind & ORT_POOL_SCALE_ERROR),
        {
            return;
        },
        php_ort_status_scaling_error_ce,
        "Scaling error occured, please review API usage");

    fci.retval = return_value;
    zend_call_function(
        &fci, &fcc);

    ort_pool_scale(&scale);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_cast_arginfo, 0, 2, ORT\\Tensor, 0)
    ZEND_ARG_TYPE_INFO(0, type, IS_LONG, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ORT\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_NAMED_FUNCTION(php_ort_math_cast)
{
    zend_long type;
    zval *tensor;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_LONG(type)
        Z_PARAM_OBJECT_OF_CLASS(tensor, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* iv = php_ort_tensor_fetch(Z_OBJ_P(tensor));

    object_init_ex(return_value,
        php_ort_tensor_transient_ce);

    php_ort_tensor_t* rv =
        php_ort_tensor_fetch(Z_OBJ_P(return_value));
    
    rv->object =
        ort_math_result_tensor(
            iv->object->shape,
            iv->object->dimensions, 
            (ONNXTensorElementDataType) type,
            "cast");
    
    ort_math_cast_buffer(
        iv->object->data,
        rv->object->data,
        iv->object->type,
        type,
        iv->object->elements);
}

/* Function table for ORT\Math namespace */
static const zend_function_entry php_ort_math_functions[] = {
    ZEND_NS_NAMED_FE("ORT\\Math", add,      php_ort_math_add,      php_ort_math_add_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", multiply, php_ort_math_multiply, php_ort_math_multiply_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", subtract, php_ort_math_subtract, php_ort_math_subtract_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", divide,   php_ort_math_divide,   php_ort_math_divide_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", sqrt,    php_ort_math_sqrt,      php_ort_math_sqrt_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", sin,     php_ort_math_sin,       php_ort_math_sin_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", cos,     php_ort_math_cos,       php_ort_math_cos_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", exp,     php_ort_math_exp,       php_ort_math_exp_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", log,     php_ort_math_log,       php_ort_math_log_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", matmul,  php_ort_math_matmul,    php_ort_math_matmul_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", asin,    php_ort_math_asin,      php_ort_math_asin_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", acos,    php_ort_math_acos,      php_ort_math_acos_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", atan,    php_ort_math_atan,      php_ort_math_atan_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", sinh,    php_ort_math_sinh,      php_ort_math_sinh_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", cosh,    php_ort_math_cosh,      php_ort_math_cosh_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", tanh,    php_ort_math_tanh,      php_ort_math_tanh_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", exp2,    php_ort_math_exp2,      php_ort_math_exp2_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", log2,    php_ort_math_log2,      php_ort_math_log2_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", log10,   php_ort_math_log10,     php_ort_math_log10_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", cbrt,    php_ort_math_cbrt,      php_ort_math_cbrt_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", ceil,    php_ort_math_ceil,      php_ort_math_ceil_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", floor,   php_ort_math_floor,     php_ort_math_floor_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", round,   php_ort_math_round,     php_ort_math_round_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", sign,    php_ort_math_sign,      php_ort_math_sign_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", tan,     php_ort_math_tan,       php_ort_math_tan_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", abs,     php_ort_math_abs,       php_ort_math_abs_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", pow,     php_ort_math_pow,       php_ort_math_pow_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", mod,     php_ort_math_mod,       php_ort_math_mod_arginfo)

    ZEND_NS_NAMED_FE("ORT\\Math", neg,     php_ort_math_neg,     php_ort_math_neg_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", recip,   php_ort_math_recip,   php_ort_math_recip_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", trunc,   php_ort_math_trunc,   php_ort_math_trunc_arginfo)

    ZEND_NS_NAMED_FE("ORT\\Math", arccos,  php_ort_math_arccos,  php_ort_math_arccos_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", arccosh, php_ort_math_arccosh, php_ort_math_arccosh_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", arcsin,  php_ort_math_arcsin,  php_ort_math_arcsin_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", arcsinh, php_ort_math_arcsinh, php_ort_math_arcsinh_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", arctan,  php_ort_math_arctan,  php_ort_math_arctan_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math", arctanh, php_ort_math_arctanh, php_ort_math_arctanh_arginfo)

    ZEND_NS_NAMED_FE("ORT\\Math\\reduce\\tensor", min,     php_ort_math_reduce_tensor_min, php_ort_math_reduce_tensor_min_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\reduce\\axis",   min,     php_ort_math_reduce_axis_min,   php_ort_math_reduce_axis_min_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\reduce\\tensor", max,     php_ort_math_reduce_tensor_max, php_ort_math_reduce_tensor_max_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\reduce\\axis",   max,     php_ort_math_reduce_axis_max,   php_ort_math_reduce_axis_max_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\reduce\\tensor", mean,    php_ort_math_reduce_tensor_mean, php_ort_math_reduce_tensor_mean_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\reduce\\axis",   mean,    php_ort_math_reduce_axis_mean,   php_ort_math_reduce_axis_mean_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\reduce\\tensor", sum,     php_ort_math_reduce_tensor_sum, php_ort_math_reduce_tensor_sum_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\reduce\\axis",   sum,     php_ort_math_reduce_axis_sum,   php_ort_math_reduce_axis_sum_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\reduce\\tensor", argmax,  php_ort_math_reduce_tensor_argmax, php_ort_math_reduce_tensor_argmax_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\reduce\\axis",   argmax,  php_ort_math_reduce_axis_argmax,   php_ort_math_reduce_axis_argmax_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\reduce\\axis",   softmax, php_ort_math_reduce_axis_softmax,   php_ort_math_reduce_axis_softmax_arginfo)

    ZEND_NS_NAMED_FE("ORT\\Math\\reduce", dot, php_ort_math_dot, php_ort_math_dot_arginfo)

    ZEND_NS_NAMED_FE("ORT\\Math", cast,      php_ort_math_cast,      php_ort_math_cast_arginfo)

    ZEND_NS_NAMED_FE("ORT\\Math",        scale,     php_ort_math_scale,     php_ort_math_scale_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\scale", cores,     php_ort_math_cores,     php_ort_math_cores_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\scale", threshold, php_ort_math_threshold, php_ort_math_threshold_arginfo)

    ZEND_NS_NAMED_FE("ORT\\Math",          backend, php_ort_math_backend, php_ort_math_backend_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\backend", cpu,     php_ort_math_backend_cpu, php_ort_math_backend_cpu_arginfo)
    ZEND_NS_NAMED_FE("ORT\\Math\\backend", gpu,     php_ort_math_backend_gpu, php_ort_math_backend_gpu_arginfo)

    ZEND_FE_END
};

/* {{{ */
ZEND_BEGIN_ARG_INFO_EX(php_ort_math_schema___construct_arginfo, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, symbol, IS_STRING, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Math_Schema, __construct)
{
    php_ort_math_schema_t *ort = php_ort_math_schema_fetch(Z_OBJ_P(getThis()));
    zend_string *symbol;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(symbol)
    ZEND_PARSE_PARAMETERS_END();

    ort->schema =
        ort_math_promotion_schema_symbol(symbol);

    php_ort_status_flow(
        !ort->schema,
        {
            RETURN_THROWS();
        },
        php_ort_status_schema_invalidsymbol_ce,
        "the symbol %s cannot be found",
        ZSTR_VAL(symbol));

    ort->symbol = zend_string_copy(symbol);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(php_ort_math_schema_getSymbol_arginfo, 0, 0, MAY_BE_STRING|MAY_BE_NULL)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Math_Schema, getSymbol)
{
    php_ort_math_schema_t *ort = php_ort_math_schema_fetch(Z_OBJ_P(getThis()));

    if (!ort->symbol) {
        return;
    }

    RETURN_STR_COPY(ort->symbol);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_math_schema_getKind_arginfo, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Math_Schema, getKind)
{
    php_ort_math_schema_t *ort =
        php_ort_math_schema_fetch(Z_OBJ_P(getThis()));

    ZEND_PARSE_PARAMETERS_NONE();

    RETURN_LONG(ort->schema->kind);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_math_schema_resolve_arginfo, 0, 1, IS_LONG, 0)
    ZEND_ARG_VARIADIC_INFO(0, types)
ZEND_END_ARG_INFO()

PHP_METHOD(ONNX_Math_Schema, resolve)
{
    php_ort_math_schema_t *ort = php_ort_math_schema_fetch(Z_OBJ_P(getThis()));
    zval *types;
    size_t count;

    ZEND_PARSE_PARAMETERS_START(1, -1)
        Z_PARAM_VARIADIC('+', types, count)
    ZEND_PARSE_PARAMETERS_END();

    if (ort->schema->kind == ORT_MATH_PROMOTION_SCHEMA_KIND_BINARY) {
        php_ort_status_flow(
            (count != 2),
            {
                RETURN_THROWS();
            },
            php_ort_status_schema_invalidarguments_ce,
            "binary schemas require two arguments for resolution, %zu given",
            count);

        php_ort_status_flow(
            (Z_TYPE(types[0]) != IS_LONG || Z_TYPE(types[1]) != IS_LONG),
            {
                RETURN_THROWS();
            },
            php_ort_status_schema_invalidarguments_ce,
            "binary schemas require both arguments to be integers");

        ONNXTensorElementDataType result =
            ort_math_promotion_resolve_binary(
                ort->schema,
                (ONNXTensorElementDataType) Z_LVAL(types[0]),
                (ONNXTensorElementDataType) Z_LVAL(types[1]));
        RETURN_LONG(result);
    } else if (ort->schema->kind == ORT_MATH_PROMOTION_SCHEMA_KIND_UNARY) {
        php_ort_status_flow(
            (count != 1),
            {
                RETURN_THROWS();
            },
            php_ort_status_schema_invalidarguments_ce,
            "unary schemas require one argument for resolution, %zu given",
            count);

        php_ort_status_flow(
            (Z_TYPE(types[0]) != IS_LONG),
            {
                RETURN_THROWS();
            },
            php_ort_status_schema_invalidarguments_ce,
            "unary schemas require an integer argument");

        ONNXTensorElementDataType result =
            ort_math_promotion_resolve_unary(
                ort->schema,
                (ONNXTensorElementDataType) Z_LVAL(types[0]));
        RETURN_LONG(result);
    } else {
        /* throw */
        RETURN_LONG(-1);
    }
}

static const zend_function_entry php_ort_math_schema_methods[] = {
    PHP_ME(ONNX_Math_Schema, __construct, php_ort_math_schema___construct_arginfo, ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Math_Schema, getSymbol,   php_ort_math_schema_getSymbol_arginfo,   ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Math_Schema, getKind,     php_ort_math_schema_getKind_arginfo,     ZEND_ACC_PUBLIC)
    PHP_ME(ONNX_Math_Schema, resolve,     php_ort_math_schema_resolve_arginfo,     ZEND_ACC_PUBLIC)
    PHP_FE_END
}; /* }}} */

PHP_MINIT_FUNCTION(ORT_MATH)
{
    ort_math_startup();

    zend_class_entry ce;

    INIT_NS_CLASS_ENTRY(ce, "ORT\\Math", "Schema", php_ort_math_schema_methods);
    php_ort_math_schema_ce = zend_register_internal_class(&ce);
    php_ort_math_schema_ce->create_object = php_ort_math_schema_create;
    php_ort_math_schema_ce->ce_flags |= ZEND_ACC_FINAL;

    zend_declare_class_constant_long(
        php_ort_math_schema_ce,
        ZEND_STRL("BINARY"),
        ORT_MATH_PROMOTION_SCHEMA_KIND_BINARY);
    zend_declare_class_constant_long(
        php_ort_math_schema_ce,
        ZEND_STRL("UNARY"),
        ORT_MATH_PROMOTION_SCHEMA_KIND_UNARY);

    memcpy(&php_ort_math_schema_handlers,
        zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    php_ort_math_schema_handlers.offset = XtOffsetOf(php_ort_math_schema_t, std);
    php_ort_math_schema_handlers.free_obj = php_ort_math_schema_free;
    php_ort_math_schema_handlers.get_debug_info = php_ort_math_schema_debug;
    php_ort_math_schema_handlers.clone_obj = NULL; // No cloning support

    zend_register_functions(NULL,
        php_ort_math_functions, NULL, MODULE_PERSISTENT);

    return SUCCESS;
}

PHP_RINIT_FUNCTION(ORT_MATH)
{
    ort_math_activate();

    return SUCCESS;
}

PHP_RSHUTDOWN_FUNCTION(ORT_MATH)
{
    ort_math_deactivate();

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_MATH)
{
    ort_math_shutdown();

    return SUCCESS;
}