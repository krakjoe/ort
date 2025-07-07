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

/* Mathematical functions in ONNX\Math namespace */

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_add_arginfo, 0, 2, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor_a, ONNX\\Tensor, 0)
    ZEND_ARG_INFO(0, tensor_b_or_scalar)
ZEND_END_ARG_INFO()

PHP_FUNCTION(add)
{
    zval *tensor_a_zv, *tensor_b_or_scalar;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(tensor_a_zv, php_ort_tensor_interface_ce)
        Z_PARAM_ZVAL(tensor_b_or_scalar)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_a_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_a_zv));
    ort_math_result_t* result = NULL;

    if (Z_TYPE_P(tensor_b_or_scalar) == IS_OBJECT && 
        instanceof_function(Z_OBJCE_P(tensor_b_or_scalar), php_ort_tensor_interface_ce)) {
        /* Tensor + Tensor */
        php_ort_tensor_t* tensor_b_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_b_or_scalar));
        result = ort_math_result_add(tensor_a_ort->object, tensor_b_ort->object);
    } else if (Z_TYPE_P(tensor_b_or_scalar) == IS_LONG || Z_TYPE_P(tensor_b_or_scalar) == IS_DOUBLE) {
        /* Tensor + Scalar */
        result = ort_math_result_add_scalar(tensor_a_ort->object, tensor_b_or_scalar);
    } else {
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,
            "add: second argument must be a Tensor or numeric value");
        return;
    }

    if (!result || !result->success) {
        return;
    }

    /* Create PHP tensor object for result */
    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_multiply_arginfo, 0, 2, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor_a, ONNX\\Tensor, 0)
    ZEND_ARG_INFO(0, tensor_b_or_scalar)
ZEND_END_ARG_INFO()

PHP_FUNCTION(multiply)
{
    zval *tensor_a_zv, *tensor_b_or_scalar;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(tensor_a_zv, php_ort_tensor_interface_ce)
        Z_PARAM_ZVAL(tensor_b_or_scalar)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_a_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_a_zv));
    ort_math_result_t* result = NULL;

    if (Z_TYPE_P(tensor_b_or_scalar) == IS_OBJECT && 
        instanceof_function(Z_OBJCE_P(tensor_b_or_scalar), php_ort_tensor_interface_ce)) {
        /* Tensor * Tensor (element-wise) */
        php_ort_tensor_t* tensor_b_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_b_or_scalar));
        result = ort_math_result_multiply(tensor_a_ort->object, tensor_b_ort->object);
    } else if (Z_TYPE_P(tensor_b_or_scalar) == IS_LONG || Z_TYPE_P(tensor_b_or_scalar) == IS_DOUBLE) {
        /* Tensor * Scalar */
        result = ort_math_result_multiply_scalar(tensor_a_ort->object, tensor_b_or_scalar);
    } else {
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,
            "multiply: second argument must be a Tensor or numeric value");
        return;
    }

    if (!result || !result->success) {
        return;
    }

    /* Create PHP tensor object for result */
    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_sqrt_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(sqrt)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_sqrt(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    /* Create PHP tensor object for result */
    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_subtract_arginfo, 0, 2, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor_a, ONNX\\Tensor, 0)
    ZEND_ARG_INFO(0, tensor_b_or_scalar)
ZEND_END_ARG_INFO()

PHP_FUNCTION(subtract)
{
    zval *tensor_a_zv, *tensor_b_or_scalar;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(tensor_a_zv, php_ort_tensor_interface_ce)
        Z_PARAM_ZVAL(tensor_b_or_scalar)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_a_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_a_zv));
    ort_math_result_t* result = NULL;

    if (Z_TYPE_P(tensor_b_or_scalar) == IS_OBJECT && 
        instanceof_function(Z_OBJCE_P(tensor_b_or_scalar), php_ort_tensor_interface_ce)) {
        /* Tensor - Tensor */
        php_ort_tensor_t* tensor_b_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_b_or_scalar));
        result = ort_math_result_subtract(tensor_a_ort->object, tensor_b_ort->object);
    } else if (Z_TYPE_P(tensor_b_or_scalar) == IS_LONG || Z_TYPE_P(tensor_b_or_scalar) == IS_DOUBLE) {
        /* Tensor - Scalar */
        result = ort_math_result_subtract_scalar(tensor_a_ort->object, tensor_b_or_scalar);
    } else {
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,
            "subtract: second argument must be a Tensor or numeric value");
        return;
    }

    if (!result || !result->success) {
        return;
    }

    /* Create PHP tensor object for result */
    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_divide_arginfo, 0, 2, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor_a, ONNX\\Tensor, 0)
    ZEND_ARG_INFO(0, tensor_b_or_scalar)
ZEND_END_ARG_INFO()

PHP_FUNCTION(divide)
{
    zval *tensor_a_zv, *tensor_b_or_scalar;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(tensor_a_zv, php_ort_tensor_interface_ce)
        Z_PARAM_ZVAL(tensor_b_or_scalar)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_a_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_a_zv));
    ort_math_result_t* result = NULL;

    if (Z_TYPE_P(tensor_b_or_scalar) == IS_OBJECT && 
        instanceof_function(Z_OBJCE_P(tensor_b_or_scalar), php_ort_tensor_interface_ce)) {
        /* Tensor / Tensor */
        php_ort_tensor_t* tensor_b_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_b_or_scalar));
        result = ort_math_result_divide(tensor_a_ort->object, tensor_b_ort->object);
    } else if (Z_TYPE_P(tensor_b_or_scalar) == IS_LONG || Z_TYPE_P(tensor_b_or_scalar) == IS_DOUBLE) {
        /* Tensor / Scalar */
        result = ort_math_result_divide_scalar(tensor_a_ort->object, tensor_b_or_scalar);
    } else {
        php_ort_status_throw(php_ort_status_math_invalidtype_ce,
            "divide: second argument must be a Tensor or numeric value");
        return;
    }

    if (!result || !result->success) {
        return;
    }

    /* Create PHP tensor object for result */
    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_sin_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(sin)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_sin(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    /* Create PHP tensor object for result */
    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_cos_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(cos)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_cos(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    /* Create PHP tensor object for result */
    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_exp_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(exp)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_exp(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    /* Create PHP tensor object for result */
    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_log_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(log)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_log(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    /* Create PHP tensor object for result */
    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
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
    
    ort_math_result_t* result = ort_math_result_matmul(matrix_a_ort->object, matrix_b_ort->object);

    if (!result || !result->success) {
        return;
    }

    /* Create PHP tensor object for result */
    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_asin_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(asin)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_asin(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_acos_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(acos)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_acos(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_atan_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(atan)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_atan(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_sinh_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(sinh)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_sinh(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_cosh_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(cosh)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_cosh(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_tanh_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(tanh)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_tanh(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_exp2_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(exp2)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_exp2(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_log2_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(log2)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_log2(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_log10_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(log10)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_log10(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_cbrt_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(cbrt)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_cbrt(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_ceil_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(ceil)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_ceil(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_floor_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(floor)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_floor(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_round_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(round)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_round(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_sign_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(sign)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_sign(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

/* Tangent function */
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_tan_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(tan)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_tan(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

/* Absolute value function */
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_abs_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(abs)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_abs(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

/* Power function */
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_pow_arginfo, 0, 2, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor_a, ONNX\\Tensor, 0)
    ZEND_ARG_INFO(0, tensor_b_or_scalar)
ZEND_END_ARG_INFO()

PHP_FUNCTION(pow)
{
    zval *tensor_a_zv, *tensor_b_or_scalar;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(tensor_a_zv, php_ort_tensor_interface_ce)
        Z_PARAM_ZVAL(tensor_b_or_scalar)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_a_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_a_zv));
    ort_math_result_t* result;

    if (Z_TYPE_P(tensor_b_or_scalar) == IS_OBJECT &&
        instanceof_function(Z_OBJCE_P(tensor_b_or_scalar), php_ort_tensor_interface_ce)) {
        php_ort_tensor_t* tensor_b_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_b_or_scalar));
        result = ort_math_result_pow(tensor_a_ort->object, tensor_b_ort->object);
    } else {
        result = ort_math_result_pow_scalar(tensor_a_ort->object, tensor_b_or_scalar);
    }

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

/* Modulo function */
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_mod_arginfo, 0, 2, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor_a, ONNX\\Tensor, 0)
    ZEND_ARG_INFO(0, tensor_b_or_scalar)
ZEND_END_ARG_INFO()

PHP_FUNCTION(mod)
{
    zval *tensor_a_zv, *tensor_b_or_scalar;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT_OF_CLASS(tensor_a_zv, php_ort_tensor_interface_ce)
        Z_PARAM_ZVAL(tensor_b_or_scalar)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_a_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_a_zv));
    ort_math_result_t* result;

    if (Z_TYPE_P(tensor_b_or_scalar) == IS_OBJECT &&
        instanceof_function(Z_OBJCE_P(tensor_b_or_scalar), php_ort_tensor_interface_ce)) {
        php_ort_tensor_t* tensor_b_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_b_or_scalar));
        result = ort_math_result_mod(tensor_a_ort->object, tensor_b_ort->object);
    } else {
        result = ort_math_result_mod_scalar(tensor_a_ort->object, tensor_b_or_scalar);
    }

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

/* Sum reduction function */
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_sum_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
    ZEND_ARG_INFO(0, axis)
    ZEND_ARG_TYPE_INFO(0, keepdims, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(sum)
{
    zval *tensor_zv, *axis = NULL;
    zend_bool keepdims = 0;

    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(axis)
        Z_PARAM_BOOL(keepdims)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_sum(tensor_ort->object, axis, keepdims);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

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
    ort_math_result_t* result = ort_math_result_dot(
        tensor_a_ort->object, tensor_b_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_neg_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(neg)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_neg(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    /* Create PHP tensor object for result */
    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

/* Reciprocal function */
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_recip_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(recip)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_recip(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
}

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(php_ort_math_trunc_arginfo, 0, 1, ONNX\\Tensor, 0)
    ZEND_ARG_OBJ_INFO(0, tensor, ONNX\\Tensor, 0)
ZEND_END_ARG_INFO()

PHP_FUNCTION(trunc)
{
    zval *tensor_zv;

    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(tensor_zv, php_ort_tensor_interface_ce)
    ZEND_PARSE_PARAMETERS_END();

    php_ort_tensor_t* tensor_ort = php_ort_tensor_fetch(Z_OBJ_P(tensor_zv));
    ort_math_result_t* result = ort_math_result_trunc(tensor_ort->object);

    if (!result || !result->success) {
        return;
    }

    object_init_ex(return_value, php_ort_tensor_transient_ce);
    php_ort_tensor_t* result_ort = php_ort_tensor_fetch(Z_OBJ_P(return_value));
    result_ort->object = result->tensor;

    ort_math_result_free(result);
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
    ZEND_NS_FE("ONNX\\Math", sum, php_ort_math_sum_arginfo)
    ZEND_NS_FE("ONNX\\Math", dot, php_ort_math_dot_arginfo)
    ZEND_NS_FE("ONNX\\Math", neg, php_ort_math_neg_arginfo)
    ZEND_NS_FE("ONNX\\Math", recip, php_ort_math_recip_arginfo)
    ZEND_NS_FE("ONNX\\Math", trunc, php_ort_math_trunc_arginfo)
    ZEND_FE_END
};

PHP_MINIT_FUNCTION(ORT_MATH)
{
    ort_math_startup();

    zend_register_functions(NULL,
        php_ort_math_functions, NULL, MODULE_PERSISTENT);

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_MATH)
{
    ort_math_shutdown();

    return SUCCESS;
}