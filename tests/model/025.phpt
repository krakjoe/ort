--TEST--
Check Model getMeta method for graph_name property
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
$graph_name = $model->getMeta("graph_name");
var_dump(is_string($graph_name) || is_null($graph_name));
?>
--EXPECT--
bool(true)
