--TEST--
Check Model cloning is denied
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
    $cloned = clone $model;
    echo "Cloning should not be allowed\n";
} catch (Error $e) {
    echo "Cloning correctly denied\n";
}
?>
--EXPECT--
Cloning correctly denied