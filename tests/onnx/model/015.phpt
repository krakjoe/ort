--TEST--
Check Model unload static method
--EXTENSIONS--
ort
--SKIPIF--
<?php
include(sprintf(
    "%s/../fixtures/skipif.php",
    dirname(__FILE__)));
?>
--FILE--
<?php
$model = new ONNX\Model("test_model",
    \sprintf("%s/../fixtures/add.onnx", dirname(__FILE__)));

var_dump(\ONNX\Model::unload("test_model"));

// Try to unload non-existent model

var_dump(\ONNX\Model::unload("nonexistent"));

// Ensure unloading actually happened

var_dump(\ONNX\Model::unload("test_model"));
?>
--EXPECT--
bool(true)
bool(false)
bool(false)