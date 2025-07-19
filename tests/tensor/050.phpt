--TEST--
Check Tensor getData method with various data types for coverage
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
// Test UINT8 type
$tensor_uint8 = new ORT\Tensor\Persistent(
    "tensor_uint8",
    [2],
    [255, 0],
    ORT\Tensor::UINT8
);
$data_uint8 = $tensor_uint8->getData();
var_dump($data_uint8);

// Test UINT16 type
$tensor_uint16 = new ORT\Tensor\Persistent(
    "tensor_uint16",
    [2],
    [65535, 0],
    ORT\Tensor::UINT16
);
$data_uint16 = $tensor_uint16->getData();
var_dump($data_uint16);

// Test UINT32 type
$tensor_uint32 = new ORT\Tensor\Persistent(
    "tensor_uint32",
    [2],
    [4294967295, 0],
    ORT\Tensor::UINT32
);
$data_uint32 = $tensor_uint32->getData();
var_dump($data_uint32);

// Test INT16 type coverage
$tensor_int16 = new ORT\Tensor\Persistent(
    "tensor_int16",
    [2, 2],
    [[1, 2], [3, 4]],
    ORT\Tensor::INT16
);

$data_int16 = $tensor_int16->getData();
var_dump($data_int16);
?>
--EXPECT--
array(2) {
  [0]=>
  int(255)
  [1]=>
  int(0)
}
array(2) {
  [0]=>
  int(65535)
  [1]=>
  int(0)
}
array(2) {
  [0]=>
  int(4294967295)
  [1]=>
  int(0)
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
