--TEST--
Check Model debug info
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

var_dump($model);
?>
--EXPECTF--
object(ONNX\Model)#%d (%d) {
  ["name"]=>
  string(10) "test_model"
  ["file"]=>
  string(%d) "%sadd.onnx"
  ["inputs"]=>
  array(2) {
    [0]=>
    string(1) "A"
    [1]=>
    string(1) "B"
  }
  ["outputs"]=>
  array(1) {
    [0]=>
    string(1) "C"
  }
}
