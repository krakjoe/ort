--TEST--
Check Tensor getData method with valid depth parameter
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
    [2, 3],
    [[1, 2, 3], [4, 5, 6]],
    ONNX\Tensor::INT32
);

// Get data starting from depth 0 (full array)
$data1 = $tensor->getData(0, 0);
var_dump($data1);

// Get data starting from depth 1 (inner arrays)
$data2 = $tensor->getData(0, 1);
var_dump($data2);
?>
--EXPECT--
array(2) {
  [0]=>
  array(3) {
    [0]=>
    int(1)
    [1]=>
    int(2)
    [2]=>
    int(3)
  }
  [1]=>
  array(3) {
    [0]=>
    int(4)
    [1]=>
    int(5)
    [2]=>
    int(6)
  }
}
array(3) {
  [0]=>
  int(1)
  [1]=>
  int(2)
  [2]=>
  int(3)
}
