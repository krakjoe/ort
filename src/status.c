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

zend_class_entry* php_ort_status_error_ce;

#ifdef ZTS
zend_class_entry* php_ort_status_safetyerror_ce;
#endif

zend_class_entry* php_ort_status_tensor_notfound_ce;
zend_class_entry* php_ort_status_tensor_invalidshape_ce;
zend_class_entry* php_ort_status_tensor_invalidtype_ce;
zend_class_entry* php_ort_status_tensor_invaliddata_ce;
zend_class_entry* php_ort_status_tensor_invalidmemory_ce;

// Mathematical operation exceptions
zend_class_entry* php_ort_status_math_error_ce;
zend_class_entry* php_ort_status_math_invalidshape_ce;
zend_class_entry* php_ort_status_math_invalidtype_ce;
zend_class_entry* php_ort_status_math_invalidbroadcast_ce;
zend_class_entry* php_ort_status_math_invalidoperation_ce;
zend_class_entry* php_ort_status_math_divisionbyzero_ce;

zend_class_entry* php_ort_status_model_invalidmemory_ce;
zend_class_entry* php_ort_status_model_invalidoptions_ce;
zend_class_entry* php_ort_status_model_invalidmodel_ce;
zend_class_entry* php_ort_status_model_invalidinput_ce;
zend_class_entry* php_ort_status_model_invalidoutput_ce;
zend_class_entry* php_ort_status_model_invalidindex_ce;
zend_class_entry* php_ort_status_model_invalidfile_ce;

zend_class_entry* php_ort_status_model_runtimeexception_ce;

PHP_MINIT_FUNCTION(ORT_STATUS) 
{
    zend_class_entry ce;

    /** ---------------------------------------------- **/
    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status", "Error", NULL);
    php_ort_status_error_ce =
        zend_register_internal_class_ex(
            &ce, zend_ce_error);

#ifdef ZTS
    /** ---------------------------------------------- **/
    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status", "SafetyError", NULL);
    php_ort_status_safetyerror_ce =
        zend_register_internal_class_ex(
            &ce, zend_ce_error);
#endif

    /** ----------------------------------------------------------------- **/
    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Tensor", "NotFound", NULL);
    php_ort_status_tensor_notfound_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Tensor", "InvalidType", NULL);
    php_ort_status_tensor_invalidtype_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Tensor", "InvalidShape", NULL);
    php_ort_status_tensor_invalidshape_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Tensor", "InvalidData", NULL);
    php_ort_status_tensor_invaliddata_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Tensor", "InvalidMemory", NULL);
    php_ort_status_tensor_invalidmemory_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    /** Mathematical operation exceptions --------------------------------- **/
    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Math", "Error", NULL);
    php_ort_status_math_error_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Math", "InvalidShape", NULL);
    php_ort_status_math_invalidshape_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_math_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Math", "InvalidType", NULL);
    php_ort_status_math_invalidtype_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_math_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Math", "InvalidBroadcast", NULL);
    php_ort_status_math_invalidbroadcast_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_math_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Math", "InvalidOperation", NULL);
    php_ort_status_math_invalidoperation_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_math_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Math", "DivisionByZero", NULL);
    php_ort_status_math_divisionbyzero_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_math_error_ce);

    /** ----------------------------------------------------------------- **/
    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Model", "InvalidMemory", NULL);
    php_ort_status_model_invalidmemory_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Model", "InvalidOptions", NULL);
    php_ort_status_model_invalidoptions_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Model", "InvalidModel", NULL);
    php_ort_status_model_invalidmodel_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Model", "InvalidInput", NULL);
    php_ort_status_model_invalidinput_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Model", "InvalidOutput", NULL);
    php_ort_status_model_invalidoutput_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Model", "InvalidIndex", NULL);
    php_ort_status_model_invalidindex_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Model", "InvalidFile", NULL);
    php_ort_status_model_invalidfile_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    INIT_NS_CLASS_ENTRY(ce, "ONNX\\Status\\Model", "RuntimeException", NULL);
    php_ort_status_model_runtimeexception_ce =
        zend_register_internal_class_ex(
            &ce, php_ort_status_error_ce);

    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(ORT_STATUS)
{
    return SUCCESS;
}