--TEST--
Check Model shared instance behavior
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
$model1 = new ONNX\Model("test_model",
    \sprintf("%s/../fixtures/add.onnx", dirname(__FILE__)));
$model2 = new ONNX\Model("test_model");

var_dump($model1->getName() === $model2->getName());
var_dump($model1->getFile() === $model2->getFile());
?>
--EXPECT--
bool(true)
bool(true)