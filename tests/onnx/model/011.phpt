--TEST--
Check Status\Model\InvalidIndex for getOutput with invalid index
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
try {
    $model->getOutput(-1);
} catch(ONNX\Status\Model\InvalidIndex $ex) {
    var_dump($ex->getMessage());
}
?>
--EXPECTF--
string(%d) "index -1 is out of range for output"