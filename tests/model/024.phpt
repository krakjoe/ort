--TEST--
Check Model getMeta method for domain property
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
$domain = $model->getMeta("domain");
var_dump(is_string($domain) || is_null($domain));
?>
--EXPECT--
bool(true)
