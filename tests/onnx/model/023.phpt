--TEST--
Check Model getMeta method for description property
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
$description = $model->getMeta("description");
var_dump(is_string($description) || is_null($description));
?>
--EXPECT--
bool(true)
