--TEST--
Check Tensor getData method with 3D tensor
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "test_tensor",
    [2, 2, 2],
    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    ORT\Tensor::INT32
);

$data = $tensor->getData();
var_dump($data);

// Test with depth 1
$data_depth1 = $tensor->getData(0, 1);
var_dump($data_depth1);

// Test with depth 2
$data_depth2 = $tensor->getData(0, 2);
var_dump($data_depth2);
?>
--EXPECT--
array(2) {
  [0]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      int(1)
      [1]=>
      int(2)
    }
    [1]=>
    array(2) {
      [0]=>
      int(3)
      [1]=>
      int(4)
    }
  }
  [1]=>
  array(2) {
    [0]=>
    array(2) {
      [0]=>
      int(5)
      [1]=>
      int(6)
    }
    [1]=>
    array(2) {
      [0]=>
      int(7)
      [1]=>
      int(8)
    }
  }
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    int(1)
    [1]=>
    int(2)
  }
  [1]=>
  array(2) {
    [0]=>
    int(3)
    [1]=>
    int(4)
  }
}
array(2) {
  [0]=>
  int(1)
  [1]=>
  int(2)
}
