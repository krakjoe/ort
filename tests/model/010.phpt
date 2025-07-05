--TEST--
Check Model getOutput with valid index
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
if ($model->getOutputCount() > 0) {
    $output = $model->getOutput(0);
    var_dump($output);
} else {
    echo "No outputs available\n";
}
?>
--EXPECTF--
string(%d) "%s"