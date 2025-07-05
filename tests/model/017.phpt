--TEST--
Check Status\Model\InvalidFile for mismatched file
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
$model1 = new ONNX\Model("mismatch_test",
    \sprintf("%s/../fixtures/add.onnx", dirname(__FILE__)));

try {
    // Try to load same name with different file
    $model2 = new ONNX\Model("mismatch_test",
        \sprintf("%s/../fixtures/sub.onnx", dirname(__FILE__)));
} catch(ONNX\Status\Model\InvalidFile $ex) {
    var_dump($ex->getMessage());
}
?>
--EXPECTF--
string(%d) "a model with the name mismatch_test is already loaded from %sadd.onnx, the file %ssub.onnx does not match"
