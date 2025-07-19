--TEST--
Check Model getMeta method for custom metadata property
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
$model = new ORT\Model("test_model",
    \sprintf("%s/../fixtures/add.onnx", dirname(__FILE__)));
$custom_property = $model->getMeta("some_custom_key");
var_dump(is_string($custom_property) || is_null($custom_property));
?>
--EXPECT--
bool(true)
