--TEST--
Check Model getMeta method for version property
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
$version = $model->getMeta("version");
var_dump(is_int($version) || is_null($version));
?>
--EXPECT--
bool(true)