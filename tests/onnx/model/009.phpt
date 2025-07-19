--TEST--
Check Status\Model\InvalidIndex for getInput with out of bounds index
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
$count = $model->getInputCount();
try {
    $model->getInput($count);
} catch(ONNX\Status\Model\InvalidIndex $ex) {
    var_dump(strpos($ex->getMessage(), 
        "is out of range for input") !== false);
}
?>
--EXPECT--
bool(true)