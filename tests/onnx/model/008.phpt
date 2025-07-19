--TEST--
Check Status\Model\InvalidIndex for getInput with invalid index
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
try {
    $model->getInput(-1);
} catch(ORT\Status\Model\InvalidIndex $ex) {
    var_dump($ex->getMessage());
}
?>
--EXPECTF--
string(%d) "index -1 is out of range for input"