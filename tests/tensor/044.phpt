--TEST--
Check Tensor getData method with 1D tensor
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
$tensor = new ONNX\Tensor\Persistent(
    "test_tensor",
    [4],
    [1, 2, 3, 4],
    ONNX\Tensor::INT32
);

$data = $tensor->getData();
var_dump($data);

// Test with depth parameter instead of offset
$data_depth = $tensor->getData(0, 0);
var_dump($data_depth);
?>
--EXPECT--
array(4) {
  [0]=>
  int(1)
  [1]=>
  int(2)
  [2]=>
  int(3)
  [3]=>
  int(4)
}
array(4) {
  [0]=>
  int(1)
  [1]=>
  int(2)
  [2]=>
  int(3)
  [3]=>
  int(4)
}
