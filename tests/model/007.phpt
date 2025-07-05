--TEST--
Check Model getInput with valid index
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
var_dump(
    $model->getInput(0));
?>
--EXPECTF--
string(%d) "%s"