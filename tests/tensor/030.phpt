--TEST--
Check Tensor handles overflow gracefully (int8)
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new \ORT\Tensor\Persistent(
    "test_tensor",
    [2, 2],
    [[127, 128], [129, 130]],
    \ORT\Tensor::INT8
);

var_dump($tensor->getData());
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(127)
    [1]=>
    int(-128)
  }
  [1]=>
  array(2) {
    [0]=>
    int(-127)
    [1]=>
    int(-126)
  }
}