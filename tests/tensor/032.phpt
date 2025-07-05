--TEST--
Check Tensor handles overflow gracefully (uint16)
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new \ONNX\Tensor\Persistent(
    "test_tensor",
    [2, 2],
    [[65535, 65536], [65537, 65538]],
    \ONNX\Tensor::UINT16
);

var_dump($tensor->getData());
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(65535)
    [1]=>
    int(0)
  }
  [1]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(2)
  }
}