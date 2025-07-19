--TEST--
Check Model getFile method
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
var_dump(\basename(
    $model->getFile()));
?>
--EXPECT--
string(8) "add.onnx"