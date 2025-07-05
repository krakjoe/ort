--TEST--
Check Runtime constructor with explicit options
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
$runtime = new \ONNX\Runtime($model, new \ONNX\Options());

echo "OK";
?>
--EXPECT--
OK