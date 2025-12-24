/* This is a generated file, edit the .stub.php file instead.
 * Stub hash: 53c3883a661aebd2dd19193025636671e84afc6f */

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_ORT_Math_add, 0, 2, ORT\\Tensor, 0)
	ZEND_ARG_OBJ_INFO(0, tensor_a, ORT\\Tensor, 0)
	ZEND_ARG_OBJ_TYPE_MASK(0, tensor_b_or_scalar, ORT\\Tensor, MAY_BE_LONG|MAY_BE_DOUBLE, NULL)
ZEND_END_ARG_INFO()

#define arginfo_ORT_Math_subtract arginfo_ORT_Math_add

#define arginfo_ORT_Math_multiply arginfo_ORT_Math_add

#define arginfo_ORT_Math_divide arginfo_ORT_Math_add

#define arginfo_ORT_Math_mod arginfo_ORT_Math_add

#define arginfo_ORT_Math_pow arginfo_ORT_Math_add

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_ORT_Math_neg, 0, 1, ORT\\Tensor, 0)
	ZEND_ARG_OBJ_INFO(0, tensor, ORT\\Tensor, 0)
ZEND_END_ARG_INFO()

#define arginfo_ORT_Math_recip arginfo_ORT_Math_neg

#define arginfo_ORT_Math_abs arginfo_ORT_Math_neg

#define arginfo_ORT_Math_sign arginfo_ORT_Math_neg

#define arginfo_ORT_Math_exp arginfo_ORT_Math_neg

#define arginfo_ORT_Math_exp2 arginfo_ORT_Math_neg

#define arginfo_ORT_Math_log arginfo_ORT_Math_neg

#define arginfo_ORT_Math_log2 arginfo_ORT_Math_neg

#define arginfo_ORT_Math_log10 arginfo_ORT_Math_neg

#define arginfo_ORT_Math_sqrt arginfo_ORT_Math_neg

#define arginfo_ORT_Math_cbrt arginfo_ORT_Math_neg

#define arginfo_ORT_Math_sin arginfo_ORT_Math_neg

#define arginfo_ORT_Math_cos arginfo_ORT_Math_neg

#define arginfo_ORT_Math_tan arginfo_ORT_Math_neg

#define arginfo_ORT_Math_asin arginfo_ORT_Math_neg

#define arginfo_ORT_Math_acos arginfo_ORT_Math_neg

#define arginfo_ORT_Math_atan arginfo_ORT_Math_neg

#define arginfo_ORT_Math_arcsin arginfo_ORT_Math_neg

#define arginfo_ORT_Math_arccos arginfo_ORT_Math_neg

#define arginfo_ORT_Math_arctan arginfo_ORT_Math_neg

#define arginfo_ORT_Math_sinh arginfo_ORT_Math_neg

#define arginfo_ORT_Math_cosh arginfo_ORT_Math_neg

#define arginfo_ORT_Math_tanh arginfo_ORT_Math_neg

#define arginfo_ORT_Math_arcsinh arginfo_ORT_Math_neg

#define arginfo_ORT_Math_arccosh arginfo_ORT_Math_neg

#define arginfo_ORT_Math_arctanh arginfo_ORT_Math_neg

#define arginfo_ORT_Math_ceil arginfo_ORT_Math_neg

#define arginfo_ORT_Math_floor arginfo_ORT_Math_neg

#define arginfo_ORT_Math_round arginfo_ORT_Math_neg

#define arginfo_ORT_Math_trunc arginfo_ORT_Math_neg

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_ORT_Math_matmul, 0, 2, ORT\\Tensor, 0)
	ZEND_ARG_OBJ_INFO(0, matrix_a, ORT\\Tensor, 0)
	ZEND_ARG_OBJ_INFO(0, matrix_b, ORT\\Tensor, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_ORT_Math_cast, 0, 2, ORT\\Tensor, 0)
	ZEND_ARG_TYPE_INFO(0, type, IS_LONG, 0)
	ZEND_ARG_OBJ_INFO(0, tensor, ORT\\Tensor, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ORT_Math_scale, 0, 2, IS_MIXED, 0)
	ZEND_ARG_TYPE_INFO(0, cores, IS_LONG, 0)
	ZEND_ARG_TYPE_INFO(0, code, IS_CALLABLE, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, threshold, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_ORT_Math_backend, 0, 0, MAY_BE_ARRAY|MAY_BE_FALSE)
ZEND_END_ARG_INFO()

#define arginfo_ORT_Math_reduce_tensor_min arginfo_ORT_Math_neg

#define arginfo_ORT_Math_reduce_tensor_max arginfo_ORT_Math_neg

#define arginfo_ORT_Math_reduce_tensor_mean arginfo_ORT_Math_neg

#define arginfo_ORT_Math_reduce_tensor_sum arginfo_ORT_Math_neg

#define arginfo_ORT_Math_reduce_tensor_argmax arginfo_ORT_Math_neg

#define arginfo_ORT_Math_reduce_tensor_argmin arginfo_ORT_Math_neg

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_ORT_Math_reduce_axis_min, 0, 2, ORT\\Tensor, 0)
	ZEND_ARG_OBJ_INFO(0, tensor, ORT\\Tensor, 0)
	ZEND_ARG_TYPE_INFO(0, axis, IS_LONG, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, keepdims, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

#define arginfo_ORT_Math_reduce_axis_max arginfo_ORT_Math_reduce_axis_min

#define arginfo_ORT_Math_reduce_axis_mean arginfo_ORT_Math_reduce_axis_min

#define arginfo_ORT_Math_reduce_axis_sum arginfo_ORT_Math_reduce_axis_min

#define arginfo_ORT_Math_reduce_axis_argmax arginfo_ORT_Math_reduce_axis_min

#define arginfo_ORT_Math_reduce_axis_argmin arginfo_ORT_Math_reduce_axis_min

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_ORT_Math_reduce_dot, 0, 2, ORT\\Tensor, 0)
	ZEND_ARG_OBJ_INFO(0, tensor_a, ORT\\Tensor, 0)
	ZEND_ARG_OBJ_INFO(0, tensor_b, ORT\\Tensor, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_ORT_Math_transform_axis_softmax, 0, 2, ORT\\Tensor, 0)
	ZEND_ARG_OBJ_INFO(0, tensor, ORT\\Tensor, 0)
	ZEND_ARG_TYPE_INFO(0, axis, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ORT_Math_scale_cores, 0, 0, IS_LONG, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, max, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ORT_Math_scale_threshold, 0, 0, IS_LONG, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, default, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(arginfo_ORT_Math_backend_cpu, 0, 0, MAY_BE_STRING|MAY_BE_FALSE)
ZEND_END_ARG_INFO()

#define arginfo_ORT_Math_backend_gpu arginfo_ORT_Math_backend_cpu

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Tensor_isPersistent, 0, 0, _IS_BOOL, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Tensor_getName, 0, 0, IS_STRING, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Tensor_getType, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Tensor_getTypeName, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Tensor_getShape, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ORT_Tensor_getSlice, 0, 2, ORT\\Tensor, 0)
	ZEND_ARG_TYPE_INFO(0, start, IS_ARRAY, 0)
	ZEND_ARG_TYPE_INFO(0, end, IS_ARRAY, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_ARRAY, 0, "[]")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Tensor_getData, 0, 0, IS_ARRAY, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, offset, IS_LONG, 0, "0")
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, depth, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ORT_Tensor_transpose, 0, 0, ORT\\Tensor, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_ARRAY, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_class_ORT_Model___construct, 0, 0, 1)
	ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, source, IS_STRING, 0, "\'\'")
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, array, _IS_BOOL, 0, "false")
	ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, options, ORT\\Options, 1, "null")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Model_unload, 0, 1, _IS_BOOL, 0)
	ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ORT_Model_getName arginfo_class_ORT_Tensor_getTypeName

#define arginfo_class_ORT_Model_getFile arginfo_class_ORT_Tensor_getTypeName

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Model_getInput, 0, 1, IS_STRING, 0)
	ZEND_ARG_TYPE_INFO(0, index, IS_LONG, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ORT_Model_getInputCount arginfo_class_ORT_Tensor_getType

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ORT_Model_getInputIterator, 0, 0, ORT\\Model\\Iterator, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ORT_Model_getOutput arginfo_class_ORT_Model_getInput

#define arginfo_class_ORT_Model_getOutputCount arginfo_class_ORT_Tensor_getType

#define arginfo_class_ORT_Model_getOutputIterator arginfo_class_ORT_Model_getInputIterator

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Model_getMeta, 0, 1, IS_MIXED, 0)
	ZEND_ARG_TYPE_INFO(0, property, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_class_ORT_Runtime___construct, 0, 0, 1)
	ZEND_ARG_OBJ_INFO(0, model, ORT\\Model, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Runtime_run, 0, 1, IS_MIXED, 0)
	ZEND_ARG_TYPE_INFO(0, input, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_class_ORT_Options___construct, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Options_setExecutionMode, 0, 1, IS_VOID, 0)
	ZEND_ARG_TYPE_INFO(0, mode, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Options_setGraphOptimization, 0, 1, IS_VOID, 0)
	ZEND_ARG_TYPE_INFO(0, optimization, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Options_setLogId, 0, 1, IS_VOID, 0)
	ZEND_ARG_TYPE_INFO(0, id, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Options_setLogSeverity, 0, 1, IS_VOID, 0)
	ZEND_ARG_TYPE_INFO(0, severity, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Options_setLogVerbosity, 0, 1, IS_VOID, 0)
	ZEND_ARG_TYPE_INFO(0, verbosity, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_class_ORT_Tensor_Persistent___construct, 0, 0, 1)
	ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, shape, IS_ARRAY, 0, "[]")
	ZEND_ARG_TYPE_MASK(0, data, MAY_BE_ARRAY|MAY_BE_OBJECT, "[]")
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, type, IS_LONG, 0, "ORT\\Tensor::FLOAT32")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ORT_Tensor_Persistent_from, 0, 3, ORT\\Tensor, 0)
	ZEND_ARG_TYPE_INFO(0, name, IS_STRING, 0)
	ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
	ZEND_ARG_TYPE_INFO(0, type, IS_LONG, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ORT_Tensor_Persistent_isPersistent arginfo_class_ORT_Tensor_isPersistent

#define arginfo_class_ORT_Tensor_Persistent_getName arginfo_class_ORT_Tensor_getName

#define arginfo_class_ORT_Tensor_Persistent_getType arginfo_class_ORT_Tensor_getType

#define arginfo_class_ORT_Tensor_Persistent_getTypeName arginfo_class_ORT_Tensor_getTypeName

#define arginfo_class_ORT_Tensor_Persistent_getShape arginfo_class_ORT_Tensor_getShape

#define arginfo_class_ORT_Tensor_Persistent_getSlice arginfo_class_ORT_Tensor_getSlice

#define arginfo_class_ORT_Tensor_Persistent_getData arginfo_class_ORT_Tensor_getData

#define arginfo_class_ORT_Tensor_Persistent_transpose arginfo_class_ORT_Tensor_transpose

ZEND_BEGIN_ARG_INFO_EX(arginfo_class_ORT_Tensor_Transient___construct, 0, 0, 2)
	ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
	ZEND_ARG_TYPE_MASK(0, data, MAY_BE_ARRAY|MAY_BE_OBJECT, NULL)
	ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, type, IS_LONG, 0, "ORT\\Tensor::FLOAT32")
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(arginfo_class_ORT_Tensor_Transient_from, 0, 2, ORT\\Tensor, 0)
	ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
	ZEND_ARG_TYPE_INFO(0, type, IS_LONG, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ORT_Tensor_Transient_isPersistent arginfo_class_ORT_Tensor_isPersistent

#define arginfo_class_ORT_Tensor_Transient_getName arginfo_class_ORT_Tensor_getName

#define arginfo_class_ORT_Tensor_Transient_getType arginfo_class_ORT_Tensor_getType

#define arginfo_class_ORT_Tensor_Transient_getTypeName arginfo_class_ORT_Tensor_getTypeName

#define arginfo_class_ORT_Tensor_Transient_getShape arginfo_class_ORT_Tensor_getShape

#define arginfo_class_ORT_Tensor_Transient_getSlice arginfo_class_ORT_Tensor_getSlice

#define arginfo_class_ORT_Tensor_Transient_getData arginfo_class_ORT_Tensor_getData

#define arginfo_class_ORT_Tensor_Transient_transpose arginfo_class_ORT_Tensor_transpose

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Tensor_Generator___invoke, 0, 0, IS_MIXED, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_class_ORT_Tensor_Generator_Random___construct, 0, 0, 1)
	ZEND_ARG_TYPE_INFO(0, type, IS_LONG, 0)
	ZEND_ARG_TYPE_MASK(0, min, MAY_BE_LONG|MAY_BE_DOUBLE, "0")
	ZEND_ARG_TYPE_MASK(0, max, MAY_BE_LONG|MAY_BE_DOUBLE, "1")
ZEND_END_ARG_INFO()

#define arginfo_class_ORT_Tensor_Generator_Random___invoke arginfo_class_ORT_Tensor_Generator___invoke

#define arginfo_class_ORT_Model_Iterator_current arginfo_class_ORT_Tensor_getTypeName

#define arginfo_class_ORT_Model_Iterator_key arginfo_class_ORT_Tensor_getType

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Model_Iterator_next, 0, 0, IS_VOID, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ORT_Model_Iterator_rewind arginfo_class_ORT_Model_Iterator_next

#define arginfo_class_ORT_Model_Iterator_valid arginfo_class_ORT_Tensor_isPersistent

ZEND_BEGIN_ARG_INFO_EX(arginfo_class_ORT_Math_Schema___construct, 0, 0, 1)
	ZEND_ARG_TYPE_INFO(0, symbol, IS_STRING, 0)
ZEND_END_ARG_INFO()

#define arginfo_class_ORT_Math_Schema_getSymbol arginfo_class_ORT_Tensor_getName

#define arginfo_class_ORT_Math_Schema_getKind arginfo_class_ORT_Tensor_getType

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_class_ORT_Math_Schema_resolve, 0, 0, IS_LONG, 0)
	ZEND_ARG_VARIADIC_TYPE_INFO(0, types, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_FUNCTION(ORT_Math_add);
ZEND_FUNCTION(ORT_Math_subtract);
ZEND_FUNCTION(ORT_Math_multiply);
ZEND_FUNCTION(ORT_Math_divide);
ZEND_FUNCTION(ORT_Math_mod);
ZEND_FUNCTION(ORT_Math_pow);
ZEND_FUNCTION(ORT_Math_neg);
ZEND_FUNCTION(ORT_Math_recip);
ZEND_FUNCTION(ORT_Math_abs);
ZEND_FUNCTION(ORT_Math_sign);
ZEND_FUNCTION(ORT_Math_exp);
ZEND_FUNCTION(ORT_Math_exp2);
ZEND_FUNCTION(ORT_Math_log);
ZEND_FUNCTION(ORT_Math_log2);
ZEND_FUNCTION(ORT_Math_log10);
ZEND_FUNCTION(ORT_Math_sqrt);
ZEND_FUNCTION(ORT_Math_cbrt);
ZEND_FUNCTION(ORT_Math_sin);
ZEND_FUNCTION(ORT_Math_cos);
ZEND_FUNCTION(ORT_Math_tan);
ZEND_FUNCTION(ORT_Math_asin);
ZEND_FUNCTION(ORT_Math_acos);
ZEND_FUNCTION(ORT_Math_atan);
ZEND_FUNCTION(ORT_Math_arcsin);
ZEND_FUNCTION(ORT_Math_arccos);
ZEND_FUNCTION(ORT_Math_arctan);
ZEND_FUNCTION(ORT_Math_sinh);
ZEND_FUNCTION(ORT_Math_cosh);
ZEND_FUNCTION(ORT_Math_tanh);
ZEND_FUNCTION(ORT_Math_arcsinh);
ZEND_FUNCTION(ORT_Math_arccosh);
ZEND_FUNCTION(ORT_Math_arctanh);
ZEND_FUNCTION(ORT_Math_ceil);
ZEND_FUNCTION(ORT_Math_floor);
ZEND_FUNCTION(ORT_Math_round);
ZEND_FUNCTION(ORT_Math_trunc);
ZEND_FUNCTION(ORT_Math_matmul);
ZEND_FUNCTION(ORT_Math_cast);
ZEND_FUNCTION(ORT_Math_scale);
ZEND_FUNCTION(ORT_Math_backend);
ZEND_FUNCTION(ORT_Math_reduce_tensor_min);
ZEND_FUNCTION(ORT_Math_reduce_tensor_max);
ZEND_FUNCTION(ORT_Math_reduce_tensor_mean);
ZEND_FUNCTION(ORT_Math_reduce_tensor_sum);
ZEND_FUNCTION(ORT_Math_reduce_tensor_argmax);
ZEND_FUNCTION(ORT_Math_reduce_tensor_argmin);
ZEND_FUNCTION(ORT_Math_reduce_axis_min);
ZEND_FUNCTION(ORT_Math_reduce_axis_max);
ZEND_FUNCTION(ORT_Math_reduce_axis_mean);
ZEND_FUNCTION(ORT_Math_reduce_axis_sum);
ZEND_FUNCTION(ORT_Math_reduce_axis_argmax);
ZEND_FUNCTION(ORT_Math_reduce_axis_argmin);
ZEND_FUNCTION(ORT_Math_reduce_dot);
ZEND_FUNCTION(ORT_Math_transform_axis_softmax);
ZEND_FUNCTION(ORT_Math_scale_cores);
ZEND_FUNCTION(ORT_Math_scale_threshold);
ZEND_FUNCTION(ORT_Math_backend_cpu);
ZEND_FUNCTION(ORT_Math_backend_gpu);
ZEND_METHOD(ORT_Model, __construct);
ZEND_METHOD(ORT_Model, unload);
ZEND_METHOD(ORT_Model, getName);
ZEND_METHOD(ORT_Model, getFile);
ZEND_METHOD(ORT_Model, getInput);
ZEND_METHOD(ORT_Model, getInputCount);
ZEND_METHOD(ORT_Model, getInputIterator);
ZEND_METHOD(ORT_Model, getOutput);
ZEND_METHOD(ORT_Model, getOutputCount);
ZEND_METHOD(ORT_Model, getOutputIterator);
ZEND_METHOD(ORT_Model, getMeta);
ZEND_METHOD(ORT_Runtime, __construct);
ZEND_METHOD(ORT_Runtime, run);
ZEND_METHOD(ORT_Options, __construct);
ZEND_METHOD(ORT_Options, setExecutionMode);
ZEND_METHOD(ORT_Options, setGraphOptimization);
ZEND_METHOD(ORT_Options, setLogId);
ZEND_METHOD(ORT_Options, setLogSeverity);
ZEND_METHOD(ORT_Options, setLogVerbosity);
ZEND_METHOD(ORT_Tensor_Persistent, __construct);
ZEND_METHOD(ORT_Tensor_Persistent, from);
ZEND_METHOD(ORT_Tensor_Persistent, isPersistent);
ZEND_METHOD(ORT_Tensor_Persistent, getName);
ZEND_METHOD(ORT_Tensor_Persistent, getType);
ZEND_METHOD(ORT_Tensor_Persistent, getTypeName);
ZEND_METHOD(ORT_Tensor_Persistent, getShape);
ZEND_METHOD(ORT_Tensor_Persistent, getSlice);
ZEND_METHOD(ORT_Tensor_Persistent, getData);
ZEND_METHOD(ORT_Tensor_Persistent, transpose);
ZEND_METHOD(ORT_Tensor_Transient, __construct);
ZEND_METHOD(ORT_Tensor_Transient, from);
ZEND_METHOD(ORT_Tensor_Transient, isPersistent);
ZEND_METHOD(ORT_Tensor_Transient, getName);
ZEND_METHOD(ORT_Tensor_Transient, getType);
ZEND_METHOD(ORT_Tensor_Transient, getTypeName);
ZEND_METHOD(ORT_Tensor_Transient, getShape);
ZEND_METHOD(ORT_Tensor_Transient, getSlice);
ZEND_METHOD(ORT_Tensor_Transient, getData);
ZEND_METHOD(ORT_Tensor_Transient, transpose);
ZEND_METHOD(ORT_Tensor_Generator_Random, __construct);
ZEND_METHOD(ORT_Tensor_Generator_Random, __invoke);
ZEND_METHOD(ORT_Model_Iterator, current);
ZEND_METHOD(ORT_Model_Iterator, key);
ZEND_METHOD(ORT_Model_Iterator, next);
ZEND_METHOD(ORT_Model_Iterator, rewind);
ZEND_METHOD(ORT_Model_Iterator, valid);
ZEND_METHOD(ORT_Math_Schema, __construct);
ZEND_METHOD(ORT_Math_Schema, getSymbol);
ZEND_METHOD(ORT_Math_Schema, getKind);
ZEND_METHOD(ORT_Math_Schema, resolve);

static const zend_function_entry ext_functions[] = {
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "add"), zif_ORT_Math_add, arginfo_ORT_Math_add, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "add"), zif_ORT_Math_add, arginfo_ORT_Math_add, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "subtract"), zif_ORT_Math_subtract, arginfo_ORT_Math_subtract, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "subtract"), zif_ORT_Math_subtract, arginfo_ORT_Math_subtract, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "multiply"), zif_ORT_Math_multiply, arginfo_ORT_Math_multiply, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "multiply"), zif_ORT_Math_multiply, arginfo_ORT_Math_multiply, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "divide"), zif_ORT_Math_divide, arginfo_ORT_Math_divide, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "divide"), zif_ORT_Math_divide, arginfo_ORT_Math_divide, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "mod"), zif_ORT_Math_mod, arginfo_ORT_Math_mod, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "mod"), zif_ORT_Math_mod, arginfo_ORT_Math_mod, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "pow"), zif_ORT_Math_pow, arginfo_ORT_Math_pow, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "pow"), zif_ORT_Math_pow, arginfo_ORT_Math_pow, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "neg"), zif_ORT_Math_neg, arginfo_ORT_Math_neg, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "neg"), zif_ORT_Math_neg, arginfo_ORT_Math_neg, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "recip"), zif_ORT_Math_recip, arginfo_ORT_Math_recip, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "recip"), zif_ORT_Math_recip, arginfo_ORT_Math_recip, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "abs"), zif_ORT_Math_abs, arginfo_ORT_Math_abs, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "abs"), zif_ORT_Math_abs, arginfo_ORT_Math_abs, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "sign"), zif_ORT_Math_sign, arginfo_ORT_Math_sign, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "sign"), zif_ORT_Math_sign, arginfo_ORT_Math_sign, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "exp"), zif_ORT_Math_exp, arginfo_ORT_Math_exp, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "exp"), zif_ORT_Math_exp, arginfo_ORT_Math_exp, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "exp2"), zif_ORT_Math_exp2, arginfo_ORT_Math_exp2, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "exp2"), zif_ORT_Math_exp2, arginfo_ORT_Math_exp2, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "log"), zif_ORT_Math_log, arginfo_ORT_Math_log, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "log"), zif_ORT_Math_log, arginfo_ORT_Math_log, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "log2"), zif_ORT_Math_log2, arginfo_ORT_Math_log2, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "log2"), zif_ORT_Math_log2, arginfo_ORT_Math_log2, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "log10"), zif_ORT_Math_log10, arginfo_ORT_Math_log10, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "log10"), zif_ORT_Math_log10, arginfo_ORT_Math_log10, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "sqrt"), zif_ORT_Math_sqrt, arginfo_ORT_Math_sqrt, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "sqrt"), zif_ORT_Math_sqrt, arginfo_ORT_Math_sqrt, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "cbrt"), zif_ORT_Math_cbrt, arginfo_ORT_Math_cbrt, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "cbrt"), zif_ORT_Math_cbrt, arginfo_ORT_Math_cbrt, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "sin"), zif_ORT_Math_sin, arginfo_ORT_Math_sin, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "sin"), zif_ORT_Math_sin, arginfo_ORT_Math_sin, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "cos"), zif_ORT_Math_cos, arginfo_ORT_Math_cos, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "cos"), zif_ORT_Math_cos, arginfo_ORT_Math_cos, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "tan"), zif_ORT_Math_tan, arginfo_ORT_Math_tan, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "tan"), zif_ORT_Math_tan, arginfo_ORT_Math_tan, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "asin"), zif_ORT_Math_asin, arginfo_ORT_Math_asin, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "asin"), zif_ORT_Math_asin, arginfo_ORT_Math_asin, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "acos"), zif_ORT_Math_acos, arginfo_ORT_Math_acos, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "acos"), zif_ORT_Math_acos, arginfo_ORT_Math_acos, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "atan"), zif_ORT_Math_atan, arginfo_ORT_Math_atan, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "atan"), zif_ORT_Math_atan, arginfo_ORT_Math_atan, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "arcsin"), zif_ORT_Math_arcsin, arginfo_ORT_Math_arcsin, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "arcsin"), zif_ORT_Math_arcsin, arginfo_ORT_Math_arcsin, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "arccos"), zif_ORT_Math_arccos, arginfo_ORT_Math_arccos, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "arccos"), zif_ORT_Math_arccos, arginfo_ORT_Math_arccos, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "arctan"), zif_ORT_Math_arctan, arginfo_ORT_Math_arctan, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "arctan"), zif_ORT_Math_arctan, arginfo_ORT_Math_arctan, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "sinh"), zif_ORT_Math_sinh, arginfo_ORT_Math_sinh, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "sinh"), zif_ORT_Math_sinh, arginfo_ORT_Math_sinh, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "cosh"), zif_ORT_Math_cosh, arginfo_ORT_Math_cosh, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "cosh"), zif_ORT_Math_cosh, arginfo_ORT_Math_cosh, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "tanh"), zif_ORT_Math_tanh, arginfo_ORT_Math_tanh, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "tanh"), zif_ORT_Math_tanh, arginfo_ORT_Math_tanh, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "arcsinh"), zif_ORT_Math_arcsinh, arginfo_ORT_Math_arcsinh, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "arcsinh"), zif_ORT_Math_arcsinh, arginfo_ORT_Math_arcsinh, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "arccosh"), zif_ORT_Math_arccosh, arginfo_ORT_Math_arccosh, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "arccosh"), zif_ORT_Math_arccosh, arginfo_ORT_Math_arccosh, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "arctanh"), zif_ORT_Math_arctanh, arginfo_ORT_Math_arctanh, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "arctanh"), zif_ORT_Math_arctanh, arginfo_ORT_Math_arctanh, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "ceil"), zif_ORT_Math_ceil, arginfo_ORT_Math_ceil, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "ceil"), zif_ORT_Math_ceil, arginfo_ORT_Math_ceil, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "floor"), zif_ORT_Math_floor, arginfo_ORT_Math_floor, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "floor"), zif_ORT_Math_floor, arginfo_ORT_Math_floor, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "round"), zif_ORT_Math_round, arginfo_ORT_Math_round, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "round"), zif_ORT_Math_round, arginfo_ORT_Math_round, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "trunc"), zif_ORT_Math_trunc, arginfo_ORT_Math_trunc, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "trunc"), zif_ORT_Math_trunc, arginfo_ORT_Math_trunc, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "matmul"), zif_ORT_Math_matmul, arginfo_ORT_Math_matmul, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "matmul"), zif_ORT_Math_matmul, arginfo_ORT_Math_matmul, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "cast"), zif_ORT_Math_cast, arginfo_ORT_Math_cast, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "cast"), zif_ORT_Math_cast, arginfo_ORT_Math_cast, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "scale"), zif_ORT_Math_scale, arginfo_ORT_Math_scale, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "scale"), zif_ORT_Math_scale, arginfo_ORT_Math_scale, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "backend"), zif_ORT_Math_backend, arginfo_ORT_Math_backend, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math", "backend"), zif_ORT_Math_backend, arginfo_ORT_Math_backend, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\tensor", "min"), zif_ORT_Math_reduce_tensor_min, arginfo_ORT_Math_reduce_tensor_min, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\tensor", "min"), zif_ORT_Math_reduce_tensor_min, arginfo_ORT_Math_reduce_tensor_min, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\tensor", "max"), zif_ORT_Math_reduce_tensor_max, arginfo_ORT_Math_reduce_tensor_max, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\tensor", "max"), zif_ORT_Math_reduce_tensor_max, arginfo_ORT_Math_reduce_tensor_max, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\tensor", "mean"), zif_ORT_Math_reduce_tensor_mean, arginfo_ORT_Math_reduce_tensor_mean, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\tensor", "mean"), zif_ORT_Math_reduce_tensor_mean, arginfo_ORT_Math_reduce_tensor_mean, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\tensor", "sum"), zif_ORT_Math_reduce_tensor_sum, arginfo_ORT_Math_reduce_tensor_sum, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\tensor", "sum"), zif_ORT_Math_reduce_tensor_sum, arginfo_ORT_Math_reduce_tensor_sum, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\tensor", "argmax"), zif_ORT_Math_reduce_tensor_argmax, arginfo_ORT_Math_reduce_tensor_argmax, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\tensor", "argmax"), zif_ORT_Math_reduce_tensor_argmax, arginfo_ORT_Math_reduce_tensor_argmax, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\tensor", "argmin"), zif_ORT_Math_reduce_tensor_argmin, arginfo_ORT_Math_reduce_tensor_argmin, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\tensor", "argmin"), zif_ORT_Math_reduce_tensor_argmin, arginfo_ORT_Math_reduce_tensor_argmin, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\axis", "min"), zif_ORT_Math_reduce_axis_min, arginfo_ORT_Math_reduce_axis_min, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\axis", "min"), zif_ORT_Math_reduce_axis_min, arginfo_ORT_Math_reduce_axis_min, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\axis", "max"), zif_ORT_Math_reduce_axis_max, arginfo_ORT_Math_reduce_axis_max, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\axis", "max"), zif_ORT_Math_reduce_axis_max, arginfo_ORT_Math_reduce_axis_max, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\axis", "mean"), zif_ORT_Math_reduce_axis_mean, arginfo_ORT_Math_reduce_axis_mean, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\axis", "mean"), zif_ORT_Math_reduce_axis_mean, arginfo_ORT_Math_reduce_axis_mean, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\axis", "sum"), zif_ORT_Math_reduce_axis_sum, arginfo_ORT_Math_reduce_axis_sum, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\axis", "sum"), zif_ORT_Math_reduce_axis_sum, arginfo_ORT_Math_reduce_axis_sum, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\axis", "argmax"), zif_ORT_Math_reduce_axis_argmax, arginfo_ORT_Math_reduce_axis_argmax, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\axis", "argmax"), zif_ORT_Math_reduce_axis_argmax, arginfo_ORT_Math_reduce_axis_argmax, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\axis", "argmin"), zif_ORT_Math_reduce_axis_argmin, arginfo_ORT_Math_reduce_axis_argmin, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce\\axis", "argmin"), zif_ORT_Math_reduce_axis_argmin, arginfo_ORT_Math_reduce_axis_argmin, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce", "dot"), zif_ORT_Math_reduce_dot, arginfo_ORT_Math_reduce_dot, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\reduce", "dot"), zif_ORT_Math_reduce_dot, arginfo_ORT_Math_reduce_dot, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\transform\\axis", "softmax"), zif_ORT_Math_transform_axis_softmax, arginfo_ORT_Math_transform_axis_softmax, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\transform\\axis", "softmax"), zif_ORT_Math_transform_axis_softmax, arginfo_ORT_Math_transform_axis_softmax, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\scale", "cores"), zif_ORT_Math_scale_cores, arginfo_ORT_Math_scale_cores, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\scale", "cores"), zif_ORT_Math_scale_cores, arginfo_ORT_Math_scale_cores, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\scale", "threshold"), zif_ORT_Math_scale_threshold, arginfo_ORT_Math_scale_threshold, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\scale", "threshold"), zif_ORT_Math_scale_threshold, arginfo_ORT_Math_scale_threshold, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\backend", "cpu"), zif_ORT_Math_backend_cpu, arginfo_ORT_Math_backend_cpu, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\backend", "cpu"), zif_ORT_Math_backend_cpu, arginfo_ORT_Math_backend_cpu, 0)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\backend", "gpu"), zif_ORT_Math_backend_gpu, arginfo_ORT_Math_backend_gpu, 0, NULL, NULL)
#else
	ZEND_RAW_FENTRY(ZEND_NS_NAME("ORT\\Math\\backend", "gpu"), zif_ORT_Math_backend_gpu, arginfo_ORT_Math_backend_gpu, 0)
#endif
	ZEND_FE_END
};

static const zend_function_entry class_ORT_Tensor_methods[] = {
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY("isPersistent", NULL, arginfo_class_ORT_Tensor_isPersistent, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT, NULL, NULL)
#else
	ZEND_RAW_FENTRY("isPersistent", NULL, arginfo_class_ORT_Tensor_isPersistent, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY("getName", NULL, arginfo_class_ORT_Tensor_getName, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT, NULL, NULL)
#else
	ZEND_RAW_FENTRY("getName", NULL, arginfo_class_ORT_Tensor_getName, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY("getType", NULL, arginfo_class_ORT_Tensor_getType, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT, NULL, NULL)
#else
	ZEND_RAW_FENTRY("getType", NULL, arginfo_class_ORT_Tensor_getType, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY("getTypeName", NULL, arginfo_class_ORT_Tensor_getTypeName, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT, NULL, NULL)
#else
	ZEND_RAW_FENTRY("getTypeName", NULL, arginfo_class_ORT_Tensor_getTypeName, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY("getShape", NULL, arginfo_class_ORT_Tensor_getShape, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT, NULL, NULL)
#else
	ZEND_RAW_FENTRY("getShape", NULL, arginfo_class_ORT_Tensor_getShape, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY("getSlice", NULL, arginfo_class_ORT_Tensor_getSlice, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT, NULL, NULL)
#else
	ZEND_RAW_FENTRY("getSlice", NULL, arginfo_class_ORT_Tensor_getSlice, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY("getData", NULL, arginfo_class_ORT_Tensor_getData, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT, NULL, NULL)
#else
	ZEND_RAW_FENTRY("getData", NULL, arginfo_class_ORT_Tensor_getData, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT)
#endif
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY("transpose", NULL, arginfo_class_ORT_Tensor_transpose, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT, NULL, NULL)
#else
	ZEND_RAW_FENTRY("transpose", NULL, arginfo_class_ORT_Tensor_transpose, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT)
#endif
	ZEND_FE_END
};

static const zend_function_entry class_ORT_Model_methods[] = {
	ZEND_ME(ORT_Model, __construct, arginfo_class_ORT_Model___construct, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model, unload, arginfo_class_ORT_Model_unload, ZEND_ACC_PUBLIC|ZEND_ACC_STATIC)
	ZEND_ME(ORT_Model, getName, arginfo_class_ORT_Model_getName, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model, getFile, arginfo_class_ORT_Model_getFile, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model, getInput, arginfo_class_ORT_Model_getInput, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model, getInputCount, arginfo_class_ORT_Model_getInputCount, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model, getInputIterator, arginfo_class_ORT_Model_getInputIterator, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model, getOutput, arginfo_class_ORT_Model_getOutput, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model, getOutputCount, arginfo_class_ORT_Model_getOutputCount, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model, getOutputIterator, arginfo_class_ORT_Model_getOutputIterator, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model, getMeta, arginfo_class_ORT_Model_getMeta, ZEND_ACC_PUBLIC)
	ZEND_FE_END
};

static const zend_function_entry class_ORT_Runtime_methods[] = {
	ZEND_ME(ORT_Runtime, __construct, arginfo_class_ORT_Runtime___construct, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Runtime, run, arginfo_class_ORT_Runtime_run, ZEND_ACC_PUBLIC)
	ZEND_FE_END
};

static const zend_function_entry class_ORT_Options_methods[] = {
	ZEND_ME(ORT_Options, __construct, arginfo_class_ORT_Options___construct, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Options, setExecutionMode, arginfo_class_ORT_Options_setExecutionMode, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Options, setGraphOptimization, arginfo_class_ORT_Options_setGraphOptimization, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Options, setLogId, arginfo_class_ORT_Options_setLogId, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Options, setLogSeverity, arginfo_class_ORT_Options_setLogSeverity, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Options, setLogVerbosity, arginfo_class_ORT_Options_setLogVerbosity, ZEND_ACC_PUBLIC)
	ZEND_FE_END
};

static const zend_function_entry class_ORT_Tensor_Persistent_methods[] = {
	ZEND_ME(ORT_Tensor_Persistent, __construct, arginfo_class_ORT_Tensor_Persistent___construct, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Persistent, from, arginfo_class_ORT_Tensor_Persistent_from, ZEND_ACC_PUBLIC|ZEND_ACC_STATIC)
	ZEND_ME(ORT_Tensor_Persistent, isPersistent, arginfo_class_ORT_Tensor_Persistent_isPersistent, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Persistent, getName, arginfo_class_ORT_Tensor_Persistent_getName, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Persistent, getType, arginfo_class_ORT_Tensor_Persistent_getType, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Persistent, getTypeName, arginfo_class_ORT_Tensor_Persistent_getTypeName, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Persistent, getShape, arginfo_class_ORT_Tensor_Persistent_getShape, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Persistent, getSlice, arginfo_class_ORT_Tensor_Persistent_getSlice, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Persistent, getData, arginfo_class_ORT_Tensor_Persistent_getData, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Persistent, transpose, arginfo_class_ORT_Tensor_Persistent_transpose, ZEND_ACC_PUBLIC)
	ZEND_FE_END
};

static const zend_function_entry class_ORT_Tensor_Transient_methods[] = {
	ZEND_ME(ORT_Tensor_Transient, __construct, arginfo_class_ORT_Tensor_Transient___construct, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Transient, from, arginfo_class_ORT_Tensor_Transient_from, ZEND_ACC_PUBLIC|ZEND_ACC_STATIC)
	ZEND_ME(ORT_Tensor_Transient, isPersistent, arginfo_class_ORT_Tensor_Transient_isPersistent, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Transient, getName, arginfo_class_ORT_Tensor_Transient_getName, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Transient, getType, arginfo_class_ORT_Tensor_Transient_getType, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Transient, getTypeName, arginfo_class_ORT_Tensor_Transient_getTypeName, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Transient, getShape, arginfo_class_ORT_Tensor_Transient_getShape, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Transient, getSlice, arginfo_class_ORT_Tensor_Transient_getSlice, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Transient, getData, arginfo_class_ORT_Tensor_Transient_getData, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Transient, transpose, arginfo_class_ORT_Tensor_Transient_transpose, ZEND_ACC_PUBLIC)
	ZEND_FE_END
};

static const zend_function_entry class_ORT_Tensor_Generator_methods[] = {
#if (PHP_VERSION_ID >= 80400)
	ZEND_RAW_FENTRY("__invoke", NULL, arginfo_class_ORT_Tensor_Generator___invoke, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT, NULL, NULL)
#else
	ZEND_RAW_FENTRY("__invoke", NULL, arginfo_class_ORT_Tensor_Generator___invoke, ZEND_ACC_PUBLIC|ZEND_ACC_ABSTRACT)
#endif
	ZEND_FE_END
};

static const zend_function_entry class_ORT_Tensor_Generator_Random_methods[] = {
	ZEND_ME(ORT_Tensor_Generator_Random, __construct, arginfo_class_ORT_Tensor_Generator_Random___construct, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Tensor_Generator_Random, __invoke, arginfo_class_ORT_Tensor_Generator_Random___invoke, ZEND_ACC_PUBLIC)
	ZEND_FE_END
};

static const zend_function_entry class_ORT_Model_Iterator_methods[] = {
	ZEND_ME(ORT_Model_Iterator, current, arginfo_class_ORT_Model_Iterator_current, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model_Iterator, key, arginfo_class_ORT_Model_Iterator_key, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model_Iterator, next, arginfo_class_ORT_Model_Iterator_next, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model_Iterator, rewind, arginfo_class_ORT_Model_Iterator_rewind, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Model_Iterator, valid, arginfo_class_ORT_Model_Iterator_valid, ZEND_ACC_PUBLIC)
	ZEND_FE_END
};

static const zend_function_entry class_ORT_Math_Schema_methods[] = {
	ZEND_ME(ORT_Math_Schema, __construct, arginfo_class_ORT_Math_Schema___construct, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Math_Schema, getSymbol, arginfo_class_ORT_Math_Schema_getSymbol, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Math_Schema, getKind, arginfo_class_ORT_Math_Schema_getKind, ZEND_ACC_PUBLIC)
	ZEND_ME(ORT_Math_Schema, resolve, arginfo_class_ORT_Math_Schema_resolve, ZEND_ACC_PUBLIC)
	ZEND_FE_END
};

static void register_ort_symbols(int module_number)
{
	REGISTER_LONG_CONSTANT("ORT\\TENSOR_UNDEFINED", ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, CONST_PERSISTENT);
}

static zend_class_entry *register_class_ORT_Tensor(void)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT", "Tensor", class_ORT_Tensor_methods);
	class_entry = zend_register_internal_interface(&ce);

	zval const_UNDEFINED_value;
	ZVAL_LONG(&const_UNDEFINED_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
	zend_string *const_UNDEFINED_name = zend_string_init_interned("UNDEFINED", sizeof("UNDEFINED") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_UNDEFINED_name, &const_UNDEFINED_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_UNDEFINED_name, true);

	zval const_FLOAT16_value;
	ZVAL_LONG(&const_FLOAT16_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
	zend_string *const_FLOAT16_name = zend_string_init_interned("FLOAT16", sizeof("FLOAT16") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_FLOAT16_name, &const_FLOAT16_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_FLOAT16_name, true);

	zval const_FLOAT32_value;
	ZVAL_LONG(&const_FLOAT32_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT32);
	zend_string *const_FLOAT32_name = zend_string_init_interned("FLOAT32", sizeof("FLOAT32") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_FLOAT32_name, &const_FLOAT32_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_FLOAT32_name, true);

	zval const_FLOAT64_value;
	ZVAL_LONG(&const_FLOAT64_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT64);
	zend_string *const_FLOAT64_name = zend_string_init_interned("FLOAT64", sizeof("FLOAT64") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_FLOAT64_name, &const_FLOAT64_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_FLOAT64_name, true);

	zval const_UINT8_value;
	ZVAL_LONG(&const_UINT8_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
	zend_string *const_UINT8_name = zend_string_init_interned("UINT8", sizeof("UINT8") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_UINT8_name, &const_UINT8_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_UINT8_name, true);

	zval const_INT8_value;
	ZVAL_LONG(&const_INT8_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
	zend_string *const_INT8_name = zend_string_init_interned("INT8", sizeof("INT8") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_INT8_name, &const_INT8_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_INT8_name, true);

	zval const_UINT16_value;
	ZVAL_LONG(&const_UINT16_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
	zend_string *const_UINT16_name = zend_string_init_interned("UINT16", sizeof("UINT16") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_UINT16_name, &const_UINT16_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_UINT16_name, true);

	zval const_INT16_value;
	ZVAL_LONG(&const_INT16_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);
	zend_string *const_INT16_name = zend_string_init_interned("INT16", sizeof("INT16") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_INT16_name, &const_INT16_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_INT16_name, true);

	zval const_UINT32_value;
	ZVAL_LONG(&const_UINT32_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32);
	zend_string *const_UINT32_name = zend_string_init_interned("UINT32", sizeof("UINT32") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_UINT32_name, &const_UINT32_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_UINT32_name, true);

	zval const_INT32_value;
	ZVAL_LONG(&const_INT32_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
	zend_string *const_INT32_name = zend_string_init_interned("INT32", sizeof("INT32") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_INT32_name, &const_INT32_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_INT32_name, true);

	zval const_INT64_value;
	ZVAL_LONG(&const_INT64_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
	zend_string *const_INT64_name = zend_string_init_interned("INT64", sizeof("INT64") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_INT64_name, &const_INT64_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_INT64_name, true);

	zval const_BOOL_value;
	ZVAL_LONG(&const_BOOL_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
	zend_string *const_BOOL_name = zend_string_init_interned("BOOL", sizeof("BOOL") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_BOOL_name, &const_BOOL_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_BOOL_name, true);

	return class_entry;
}

static zend_class_entry *register_class_ORT_Model(void)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT", "Model", class_ORT_Model_methods);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, NULL, ZEND_ACC_NOT_SERIALIZABLE);
#else
	class_entry = zend_register_internal_class_ex(&ce, NULL);
#if (PHP_VERSION_ID >= 80100)
	class_entry->ce_flags |= ZEND_ACC_NOT_SERIALIZABLE;
#elif (PHP_VERSION_ID >= 80000)
	class_entry->ce_flags |= 0;
#endif
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Runtime(void)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT", "Runtime", class_ORT_Runtime_methods);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, NULL, ZEND_ACC_NOT_SERIALIZABLE);
#else
	class_entry = zend_register_internal_class_ex(&ce, NULL);
#if (PHP_VERSION_ID >= 80100)
	class_entry->ce_flags |= ZEND_ACC_NOT_SERIALIZABLE;
#elif (PHP_VERSION_ID >= 80000)
	class_entry->ce_flags |= 0;
#endif
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Options(void)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT", "Options", class_ORT_Options_methods);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, NULL, ZEND_ACC_NOT_SERIALIZABLE);
#else
	class_entry = zend_register_internal_class_ex(&ce, NULL);
#if (PHP_VERSION_ID >= 80100)
	class_entry->ce_flags |= ZEND_ACC_NOT_SERIALIZABLE;
#elif (PHP_VERSION_ID >= 80000)
	class_entry->ce_flags |= 0;
#endif
#endif

	zval const_LVERBOSE_value;
	ZVAL_LONG(&const_LVERBOSE_value, ORT_LOGGING_LEVEL_VERBOSE);
	zend_string *const_LVERBOSE_name = zend_string_init_interned("LVERBOSE", sizeof("LVERBOSE") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_LVERBOSE_name, &const_LVERBOSE_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_LVERBOSE_name, true);

	zval const_LINFO_value;
	ZVAL_LONG(&const_LINFO_value, ORT_LOGGING_LEVEL_INFO);
	zend_string *const_LINFO_name = zend_string_init_interned("LINFO", sizeof("LINFO") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_LINFO_name, &const_LINFO_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_LINFO_name, true);

	zval const_LWARNING_value;
	ZVAL_LONG(&const_LWARNING_value, ORT_LOGGING_LEVEL_WARNING);
	zend_string *const_LWARNING_name = zend_string_init_interned("LWARNING", sizeof("LWARNING") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_LWARNING_name, &const_LWARNING_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_LWARNING_name, true);

	zval const_LERROR_value;
	ZVAL_LONG(&const_LERROR_value, ORT_LOGGING_LEVEL_ERROR);
	zend_string *const_LERROR_name = zend_string_init_interned("LERROR", sizeof("LERROR") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_LERROR_name, &const_LERROR_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_LERROR_name, true);

	zval const_LFATAL_value;
	ZVAL_LONG(&const_LFATAL_value, ORT_LOGGING_LEVEL_FATAL);
	zend_string *const_LFATAL_name = zend_string_init_interned("LFATAL", sizeof("LFATAL") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_LFATAL_name, &const_LFATAL_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_LFATAL_name, true);

	zval const_ESEQUENTIAL_value;
	ZVAL_LONG(&const_ESEQUENTIAL_value, ORT_SEQUENTIAL);
	zend_string *const_ESEQUENTIAL_name = zend_string_init_interned("ESEQUENTIAL", sizeof("ESEQUENTIAL") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_ESEQUENTIAL_name, &const_ESEQUENTIAL_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_ESEQUENTIAL_name, true);

	zval const_EPARALLEL_value;
	ZVAL_LONG(&const_EPARALLEL_value, ORT_PARALLEL);
	zend_string *const_EPARALLEL_name = zend_string_init_interned("EPARALLEL", sizeof("EPARALLEL") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_EPARALLEL_name, &const_EPARALLEL_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_EPARALLEL_name, true);

	zval const_ONONE_value;
	ZVAL_LONG(&const_ONONE_value, ORT_DISABLE_ALL);
	zend_string *const_ONONE_name = zend_string_init_interned("ONONE", sizeof("ONONE") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_ONONE_name, &const_ONONE_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_ONONE_name, true);

	zval const_OBASIC_value;
	ZVAL_LONG(&const_OBASIC_value, ORT_ENABLE_BASIC);
	zend_string *const_OBASIC_name = zend_string_init_interned("OBASIC", sizeof("OBASIC") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_OBASIC_name, &const_OBASIC_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_OBASIC_name, true);

	zval const_OEXTENDED_value;
	ZVAL_LONG(&const_OEXTENDED_value, ORT_ENABLE_EXTENDED);
	zend_string *const_OEXTENDED_name = zend_string_init_interned("OEXTENDED", sizeof("OEXTENDED") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_OEXTENDED_name, &const_OEXTENDED_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_OEXTENDED_name, true);

	zval const_OALL_value;
	ZVAL_LONG(&const_OALL_value, ORT_ENABLE_ALL);
	zend_string *const_OALL_name = zend_string_init_interned("OALL", sizeof("OALL") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_OALL_name, &const_OALL_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_OALL_name, true);

	return class_entry;
}

static zend_class_entry *register_class_ORT_Tensor_Persistent(zend_class_entry *class_entry_ORT_Tensor)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Tensor", "Persistent", class_ORT_Tensor_Persistent_methods);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, NULL, ZEND_ACC_NOT_SERIALIZABLE);
#else
	class_entry = zend_register_internal_class_ex(&ce, NULL);
#if (PHP_VERSION_ID >= 80100)
	class_entry->ce_flags |= ZEND_ACC_NOT_SERIALIZABLE;
#elif (PHP_VERSION_ID >= 80000)
	class_entry->ce_flags |= 0;
#endif
#endif
	zend_class_implements(class_entry, 1, class_entry_ORT_Tensor);

	return class_entry;
}

static zend_class_entry *register_class_ORT_Tensor_Transient(zend_class_entry *class_entry_ORT_Tensor)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Tensor", "Transient", class_ORT_Tensor_Transient_methods);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, NULL, ZEND_ACC_NOT_SERIALIZABLE);
#else
	class_entry = zend_register_internal_class_ex(&ce, NULL);
#if (PHP_VERSION_ID >= 80100)
	class_entry->ce_flags |= ZEND_ACC_NOT_SERIALIZABLE;
#elif (PHP_VERSION_ID >= 80000)
	class_entry->ce_flags |= 0;
#endif
#endif
	zend_class_implements(class_entry, 1, class_entry_ORT_Tensor);

	return class_entry;
}

static zend_class_entry *register_class_ORT_Tensor_Generator(void)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Tensor", "Generator", class_ORT_Tensor_Generator_methods);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, NULL, ZEND_ACC_ABSTRACT);
#else
	class_entry = zend_register_internal_class_ex(&ce, NULL);
	class_entry->ce_flags |= ZEND_ACC_ABSTRACT;
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Tensor_Generator_Random(zend_class_entry *class_entry_ORT_Tensor_Generator)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Tensor\\Generator", "Random", class_ORT_Tensor_Generator_Random_methods);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Tensor_Generator, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Tensor_Generator);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Model_Iterator(zend_class_entry *class_entry_Iterator)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Model", "Iterator", class_ORT_Model_Iterator_methods);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, NULL, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, NULL);
#endif
	zend_class_implements(class_entry, 1, class_entry_Iterator);

	return class_entry;
}

static zend_class_entry *register_class_ORT_Math_Schema(void)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Math", "Schema", class_ORT_Math_Schema_methods);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, NULL, ZEND_ACC_FINAL);
#else
	class_entry = zend_register_internal_class_ex(&ce, NULL);
	class_entry->ce_flags |= ZEND_ACC_FINAL;
#endif

	zval const_BINARY_value;
	ZVAL_LONG(&const_BINARY_value, ORT_MATH_PROMOTION_SCHEMA_KIND_BINARY);
	zend_string *const_BINARY_name = zend_string_init_interned("BINARY", sizeof("BINARY") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_BINARY_name, &const_BINARY_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_BINARY_name, true);

	zval const_UNARY_value;
	ZVAL_LONG(&const_UNARY_value, ORT_MATH_PROMOTION_SCHEMA_KIND_UNARY);
	zend_string *const_UNARY_name = zend_string_init_interned("UNARY", sizeof("UNARY") - 1, true);
	zend_declare_class_constant_ex(class_entry, const_UNARY_name, &const_UNARY_value, ZEND_ACC_PUBLIC, NULL);
	zend_string_release_ex(const_UNARY_name, true);

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Error(zend_class_entry *class_entry_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status", "Error", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_SafetyError(zend_class_entry *class_entry_ORT_Status_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status", "SafetyError", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_ScalingError(zend_class_entry *class_entry_ORT_Status_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status", "ScalingError", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Tensor_Error(zend_class_entry *class_entry_ORT_Status_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Tensor", "Error", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Tensor_NotFound(zend_class_entry *class_entry_ORT_Status_Tensor_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Tensor", "NotFound", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Tensor_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Tensor_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Tensor_InvalidType(zend_class_entry *class_entry_ORT_Status_Tensor_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Tensor", "InvalidType", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Tensor_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Tensor_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Tensor_InvalidShape(zend_class_entry *class_entry_ORT_Status_Tensor_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Tensor", "InvalidShape", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Tensor_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Tensor_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Tensor_InvalidData(zend_class_entry *class_entry_ORT_Status_Tensor_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Tensor", "InvalidData", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Tensor_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Tensor_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Tensor_InvalidMemory(zend_class_entry *class_entry_ORT_Status_Tensor_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Tensor", "InvalidMemory", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Tensor_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Tensor_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Tensor_InvalidOffset(zend_class_entry *class_entry_ORT_Status_Tensor_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Tensor", "InvalidOffset", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Tensor_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Tensor_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Tensor_InvalidAccess(zend_class_entry *class_entry_ORT_Status_Tensor_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Tensor", "InvalidAccess", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Tensor_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Tensor_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Math_Error(zend_class_entry *class_entry_ORT_Status_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Math", "Error", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Math_InvalidShape(zend_class_entry *class_entry_ORT_Status_Math_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Math", "InvalidShape", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Math_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Math_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Math_InvalidType(zend_class_entry *class_entry_ORT_Status_Math_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Math", "InvalidType", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Math_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Math_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Math_InvalidBroadcast(zend_class_entry *class_entry_ORT_Status_Math_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Math", "InvalidBroadcast", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Math_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Math_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Math_InvalidOperation(zend_class_entry *class_entry_ORT_Status_Math_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Math", "InvalidOperation", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Math_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Math_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Math_DivisionByZero(zend_class_entry *class_entry_ORT_Status_Math_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Math", "DivisionByZero", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Math_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Math_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Model_Error(zend_class_entry *class_entry_ORT_Status_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Model", "Error", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Model_InvalidMemory(zend_class_entry *class_entry_ORT_Status_Model_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Model", "InvalidMemory", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Model_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Model_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Model_InvalidOptions(zend_class_entry *class_entry_ORT_Status_Model_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Model", "InvalidOptions", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Model_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Model_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Model_InvalidModel(zend_class_entry *class_entry_ORT_Status_Model_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Model", "InvalidModel", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Model_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Model_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Model_InvalidInput(zend_class_entry *class_entry_ORT_Status_Model_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Model", "InvalidInput", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Model_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Model_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Model_InvalidOutput(zend_class_entry *class_entry_ORT_Status_Model_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Model", "InvalidOutput", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Model_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Model_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Model_InvalidIndex(zend_class_entry *class_entry_ORT_Status_Model_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Model", "InvalidIndex", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Model_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Model_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Model_InvalidFile(zend_class_entry *class_entry_ORT_Status_Model_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Model", "InvalidFile", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Model_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Model_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Model_RuntimeException(zend_class_entry *class_entry_ORT_Status_Model_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Model", "RuntimeException", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Model_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Model_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Schema_Error(zend_class_entry *class_entry_ORT_Status_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Schema", "Error", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Schema_InvalidSymbol(zend_class_entry *class_entry_ORT_Status_Schema_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Schema", "InvalidSymbol", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Schema_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Schema_Error);
#endif

	return class_entry;
}

static zend_class_entry *register_class_ORT_Status_Schema_InvalidArguments(zend_class_entry *class_entry_ORT_Status_Schema_Error)
{
	zend_class_entry ce, *class_entry;

	INIT_NS_CLASS_ENTRY(ce, "ORT\\Status\\Schema", "InvalidArguments", NULL);
#if (PHP_VERSION_ID >= 80400)
	class_entry = zend_register_internal_class_with_flags(&ce, class_entry_ORT_Status_Schema_Error, 0);
#else
	class_entry = zend_register_internal_class_ex(&ce, class_entry_ORT_Status_Schema_Error);
#endif

	return class_entry;
}
