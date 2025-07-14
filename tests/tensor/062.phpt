--TEST--
ONNX\Tensor::transpose
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = \ONNX\Tensor\Transient::from([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], \ONNX\Tensor::INT8);

var_dump($tensor->getData());

$result =
    $tensor->transpose();

var_dump($result->getData());

$result =
    $tensor->transpose([1, 0]);

var_dump($result->getData());

$result =
    $tensor->transpose([0, 1]);

var_dump($result->getData());

$result =
    $tensor->transpose([-1, -2]);

var_dump($result->getData());
?>
--EXPECT--
array(3) {
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
  [2]=>
  array(3) {
    [0]=>
    int(7)
    [1]=>
    int(8)
    [2]=>
    int(9)
  }
}
array(3) {
  [0]=>
  array(3) {
    [0]=>
    int(1)
    [1]=>
    int(4)
    [2]=>
    int(7)
  }
  [1]=>
  array(3) {
    [0]=>
    int(2)
    [1]=>
    int(5)
    [2]=>
    int(8)
  }
  [2]=>
  array(3) {
    [0]=>
    int(3)
    [1]=>
    int(6)
    [2]=>
    int(9)
  }
}
array(3) {
  [0]=>
  array(3) {
    [0]=>
    int(1)
    [1]=>
    int(4)
    [2]=>
    int(7)
  }
  [1]=>
  array(3) {
    [0]=>
    int(2)
    [1]=>
    int(5)
    [2]=>
    int(8)
  }
  [2]=>
  array(3) {
    [0]=>
    int(3)
    [1]=>
    int(6)
    [2]=>
    int(9)
  }
}
array(3) {
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
  [2]=>
  array(3) {
    [0]=>
    int(7)
    [1]=>
    int(8)
    [2]=>
    int(9)
  }
}
array(3) {
  [0]=>
  array(3) {
    [0]=>
    int(1)
    [1]=>
    int(4)
    [2]=>
    int(7)
  }
  [1]=>
  array(3) {
    [0]=>
    int(2)
    [1]=>
    int(5)
    [2]=>
    int(8)
  }
  [2]=>
  array(3) {
    [0]=>
    int(3)
    [1]=>
    int(6)
    [2]=>
    int(9)
  }
}