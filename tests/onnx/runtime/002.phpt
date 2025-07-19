--TEST--
Check Runtime constructor with explicit options
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
$runtime = new \ORT\Runtime($model, new \ORT\Options());

echo "OK";
?>
--EXPECT--
OK