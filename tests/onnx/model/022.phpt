--TEST--
Check Model getMeta method for producer property
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
$producer = $model->getMeta("producer");
var_dump(is_string($producer) || is_null($producer));
?>
--EXPECT--
bool(true)
