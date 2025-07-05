--TEST--
Check Tensor handles overflow gracefully (uint32)
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new \ONNX\Tensor\Persistent(
    "test_tensor",
    [2, 2],
    [[4294967295, 4294967296], [4294967297, 4294967298]],
    \ONNX\Tensor::UINT32
);

var_dump($tensor->getData());
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(4294967295)
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