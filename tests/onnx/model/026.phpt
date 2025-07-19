--TEST--
Check Model getMeta method for graph_description property
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
$graph_description = $model->getMeta("graph_description");
var_dump(is_string($graph_description) || is_null($graph_description));
?>
--EXPECT--
bool(true)
