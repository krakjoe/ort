--TEST--
Check Tensor handles overflow gracefully (int16)
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new \ONNX\Tensor\Persistent(
    "test_tensor",
    [2, 2],
    [[32767, 32768], [32769, 32770]],
    \ONNX\Tensor::INT16
);

var_dump($tensor->getData());
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(32767)
    [1]=>
    int(-32768)
  }
  [1]=>
  array(2) {
    [0]=>
    int(-32767)
    [1]=>
    int(-32766)
  }
}