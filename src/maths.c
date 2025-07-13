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
#include "maths/result.h"
#include "maths/promotion.h"

#include "maths/schema/add.h"
#include "maths/schema/div.h"
#include "maths/schema/dot.h"
#include "maths/schema/matmul.h"
#include "maths/schema/mod.h"
#include "maths/schema/mul.h"
#include "maths/schema/neg.h"
#include "maths/schema/recip.h"
#include "maths/schema/sign.h"
#include "maths/schema/sqrt.h"
#include "maths/schema/sub.h"

typedef struct _php_ort_math_schema_t {
    const ort_math_type_promotion_schema_t* schema;
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

__php_ort_tensor_debug_return:
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
        php_ort_math_##fname##_arginfo, 0, 1, ONNX\\Tensor, 0) \
        ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)          \
    ZEND_END_ARG_INFO()                                        \
    PHP_FUNCTION(fname)                                        \
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
        php_ort_math_##fname##_arginfo, 0, 2, ONNX\\Tensor, 0)  \
        ZEND_ARG_OBJ_INFO(0, tensor_a, ONNX\\Tensor, 0)         \
        ZEND_ARG_INFO(0, tensor_b_or_scalar)                    \
    ZEND_END_ARG_INFO()                                         \
    PHP_FUNCTION(fname)                                         \
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
                0, 1, ONNX\\Tensor, 0)                             \
        ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)              \
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
            php_ort_math_reduce_axis_##fname##_arginfo, 0, 2, ONNX\\Tensor, 0) \
        ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)              \
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
/* Mathematical functions in ONNX\Math namespace */

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

/* Dot reduction function */
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_dot_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor_a, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor_b, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(dot)
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

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_matmul_arginfo, 0, 2, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, matrix_a, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, matrix_b, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(matmul)
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

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(php_ort_math_backend_arginfo, 0, 0, MAY_BE_STRING|MAY_BE_FALSE)
ZEND_END_ARG_INFO()

PHP_FUNCTION(backend)
{
    ZEND_PARSE_PARAMETERS_NONE();

#ifdef ORT_BACKEND_ENABLED
    RETURN_STRING(ORT_BACKEND_NAME);
#else
    RETURN_FALSE;
#endif
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_math_cores_arginfo, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(cores)
{
    ZEND_PARSE_PARAMETERS_NONE();

    RETURN_LONG(
        ort_pool_cores());
}

/* Function table for ONNX\Math namespace */
static const zend_function_entry php_ort_math_functions[] = {
    ZEND_NS_FE("ONNX\\Math", add, php_ort_math_add_arginfo)
    ZEND_NS_FE("ONNX\\Math", multiply, php_ort_math_multiply_arginfo) 
    ZEND_NS_FE("ONNX\\Math", subtract, php_ort_math_subtract_arginfo)
    ZEND_NS_FE("ONNX\\Math", divide, php_ort_math_divide_arginfo)
    ZEND_NS_FE("ONNX\\Math", sqrt, php_ort_math_sqrt_arginfo)
    ZEND_NS_FE("ONNX\\Math", sin, php_ort_math_sin_arginfo)
    ZEND_NS_FE("ONNX\\Math", cos, php_ort_math_cos_arginfo)
    ZEND_NS_FE("ONNX\\Math", exp, php_ort_math_exp_arginfo)
    ZEND_NS_FE("ONNX\\Math", log, php_ort_math_log_arginfo)
    ZEND_NS_FE("ONNX\\Math", matmul, php_ort_math_matmul_arginfo)
    ZEND_NS_FE("ONNX\\Math", asin, php_ort_math_asin_arginfo)
    ZEND_NS_FE("ONNX\\Math", acos, php_ort_math_acos_arginfo)
    ZEND_NS_FE("ONNX\\Math", atan, php_ort_math_atan_arginfo)
    ZEND_NS_FE("ONNX\\Math", sinh, php_ort_math_sinh_arginfo)
    ZEND_NS_FE("ONNX\\Math", cosh, php_ort_math_cosh_arginfo)
    ZEND_NS_FE("ONNX\\Math", tanh, php_ort_math_tanh_arginfo)
    ZEND_NS_FE("ONNX\\Math", exp2, php_ort_math_exp2_arginfo)
    ZEND_NS_FE("ONNX\\Math", log2, php_ort_math_log2_arginfo)
    ZEND_NS_FE("ONNX\\Math", log10, php_ort_math_log10_arginfo)
    ZEND_NS_FE("ONNX\\Math", cbrt, php_ort_math_cbrt_arginfo)
    ZEND_NS_FE("ONNX\\Math", ceil, php_ort_math_ceil_arginfo)
    ZEND_NS_FE("ONNX\\Math", floor, php_ort_math_floor_arginfo)
    ZEND_NS_FE("ONNX\\Math", round, php_ort_math_round_arginfo)
    ZEND_NS_FE("ONNX\\Math", sign, php_ort_math_sign_arginfo)
    ZEND_NS_FE("ONNX\\Math", tan, php_ort_math_tan_arginfo)
    ZEND_NS_FE("ONNX\\Math", abs, php_ort_math_abs_arginfo)
    ZEND_NS_FE("ONNX\\Math", pow, php_ort_math_pow_arginfo)
    ZEND_NS_FE("ONNX\\Math", mod, php_ort_math_mod_arginfo)

    ZEND_NS_FE("ONNX\\Math", neg,     php_ort_math_neg_arginfo)
    ZEND_NS_FE("ONNX\\Math", recip,   php_ort_math_recip_arginfo)
    ZEND_NS_FE("ONNX\\Math", trunc,   php_ort_math_trunc_arginfo)

    ZEND_NS_NAMED_FE("ONNX\\Math\\reduce\\tensor", min, php_ort_math_reduce_tensor_min, php_ort_math_reduce_tensor_min_arginfo)
    ZEND_NS_NAMED_FE("ONNX\\Math\\reduce\\axis",   min, php_ort_math_reduce_axis_min,   php_ort_math_reduce_axis_min_arginfo)
    ZEND_NS_NAMED_FE("ONNX\\Math\\reduce\\tensor", max, php_ort_math_reduce_tensor_max, php_ort_math_reduce_tensor_max_arginfo)
    ZEND_NS_NAMED_FE("ONNX\\Math\\reduce\\axis",   max, php_ort_math_reduce_axis_max,   php_ort_math_reduce_axis_max_arginfo)
    ZEND_NS_NAMED_FE("ONNX\\Math\\reduce\\tensor", mean, php_ort_math_reduce_tensor_mean, php_ort_math_reduce_tensor_mean_arginfo)
    ZEND_NS_NAMED_FE("ONNX\\Math\\reduce\\axis",   mean, php_ort_math_reduce_axis_mean,   php_ort_math_reduce_axis_mean_arginfo)
    ZEND_NS_NAMED_FE("ONNX\\Math\\reduce\\tensor", sum, php_ort_math_reduce_tensor_sum, php_ort_math_reduce_tensor_sum_arginfo)
    ZEND_NS_NAMED_FE("ONNX\\Math\\reduce\\axis",   sum, php_ort_math_reduce_axis_sum,   php_ort_math_reduce_axis_sum_arginfo)
    ZEND_NS_NAMED_FE("ONNX\\Math\\reduce\\axis",   softmax, php_ort_math_reduce_axis_softmax,   php_ort_math_reduce_axis_softmax_arginfo)

    ZEND_NS_FE("ONNX\\Math", dot, php_ort_math_dot_arginfo)

    ZEND_NS_FE("ONNX\\Math", backend, php_ort_math_backend_arginfo)
    ZEND_NS_FE("ONNX\\Math", cores,   php_ort_math_cores_arginfo)
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

    if (zend_string_equals_literal_ci(symbol, "add")) {
        ort->schema = &ort_math_promotion_schema_add;
    } else if (zend_string_equals_literal_ci(symbol, "div")) {
        ort->schema = &ort_math_promotion_schema_div;
    } else if (zend_string_equals_literal_ci(symbol, "dot")) {
        ort->schema = &ort_math_promotion_schema_dot;
    } else if (zend_string_equals_literal_ci(symbol, "matmul")) {
        ort->schema = &ort_math_promotion_schema_matmul;
    } else if (zend_string_equals_literal_ci(symbol, "mod")) {
        ort->schema = &ort_math_promotion_schema_mod;
    } else if (zend_string_equals_literal_ci(symbol, "mul")) {
        ort->schema = &ort_math_promotion_schema_mul;
    } else if (zend_string_equals_literal_ci(symbol, "neg")) {
        ort->schema = &ort_math_promotion_schema_neg;
    } else if (zend_string_equals_literal_ci(symbol, "recip")) {
        ort->schema = &ort_math_promotion_schema_recip;
    } else if (zend_string_equals_literal_ci(symbol, "sign")) {
        ort->schema = &ort_math_promotion_schema_sign;
    } else if (zend_string_equals_literal_ci(symbol, "sqrt")) {
        ort->schema = &ort_math_promotion_schema_sqrt;
    } else if (zend_string_equals_literal_ci(symbol, "sub")) {
        ort->schema = &ort_math_promotion_schema_sub;
    } else {
        /* throw */
        return;
    }

    ort->symbol = zend_string_copy(symbol);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(php_ort_math_schema_getSymbol_arginfo, 0, 0, IS_STRING, 0)
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

    if (count == 0) {
        /* throw */
        RETURN_LONG(-1);
    }

    if (ort->schema->kind == ORT_MATH_TYPE_PROMOTION_SCHEMA_BINARY) {
        if (count != 2) {
            /* throw */
            RETURN_LONG(-1);
        }

        if (Z_TYPE(types[0]) != IS_LONG || Z_TYPE(types[1]) != IS_LONG) {
            /* throw */
            RETURN_LONG(-1);
        }

        ONNXTensorElementDataType result =
            ort_math_type_promotion_schema_resolve_binary(
                ort->schema,
                (ONNXTensorElementDataType) Z_LVAL(types[0]),
                (ONNXTensorElementDataType) Z_LVAL(types[1]));
        RETURN_LONG(result);
    } else if (ort->schema->kind == ORT_MATH_TYPE_PROMOTION_SCHEMA_UNARY) {
        if (count != 1) {
            RETURN_LONG(-1);
        }

        if (Z_TYPE(types[0]) != IS_LONG) {
            /* throw */
            RETURN_LONG(-1);
        }

        ONNXTensorElementDataType result =
            ort_math_type_promotion_schema_resolve_unary(
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

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Math", "Schema", php_ort_math_schema_methods);
    php_ort_math_schema_ce = zend_register_internal_class(&ce);
    php_ort_math_schema_ce->create_object = php_ort_math_schema_create;
    php_ort_math_schema_ce->ce_flags |= ZEND_ACC_FINAL;

    zend_declare_class_constant_long(
        php_ort_math_schema_ce,
        ZEND_STRL("BINARY"),
        ORT_MATH_TYPE_PROMOTION_SCHEMA_BINARY);
    zend_declare_class_constant_long(
        php_ort_math_schema_ce,
        ZEND_STRL("UNARY"),
        ORT_MATH_TYPE_PROMOTION_SCHEMA_UNARY);

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