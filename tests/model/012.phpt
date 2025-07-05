--TEST--
Check Model getInputIterator method
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
$model = new ONNX\Model("test_model",
    \sprintf("%s/../fixtures/add.onnx", dirname(__FILE__)));
$iterator = $model->getInputIterator();
var_dump($iterator instanceof ONNX\Model\Iterator);
?>
--EXPECT--
bool(true)