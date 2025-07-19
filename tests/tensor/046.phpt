--TEST--
Check Tensor getData method with all integer types
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
// Test INT8
$tensor_int8 = new ORT\Tensor\Persistent(
    "test_tensor_int8",
    [2],
    [127, -128],
    ORT\Tensor::INT8
);
$data_int8 = $tensor_int8->getData();
var_dump($data_int8);

// Test INT16
$tensor_int16 = new ORT\Tensor\Persistent(
    "test_tensor_int16",
    [2],
    [32767, -32768],
    ORT\Tensor::INT16
);
$data_int16 = $tensor_int16->getData();
var_dump($data_int16);

// Test INT64
$tensor_int64 = new ORT\Tensor\Persistent(
    "test_tensor_int64",
    [2],
    [1000000000000, -1000000000000],
    ORT\Tensor::INT64
);
$data_int64 = $tensor_int64->getData();
var_dump($data_int64);

// Test UINT8
$tensor_uint8 = new ORT\Tensor\Persistent(
    "test_tensor_uint8",
    [2],
    [255, 0],
    ORT\Tensor::UINT8
);
$data_uint8 = $tensor_uint8->getData();
var_dump($data_uint8);

// Test UINT16
$tensor_uint16 = new ORT\Tensor\Persistent(
    "test_tensor_uint16",
    [2],
    [65535, 0],
    ORT\Tensor::UINT16
);
$data_uint16 = $tensor_uint16->getData();
var_dump($data_uint16);

// Test UINT32
$tensor_uint32 = new ORT\Tensor\Persistent(
    "test_tensor_uint32",
    [2],
    [4294967295, 0],
    ORT\Tensor::UINT32
);
$data_uint32 = $tensor_uint32->getData();
var_dump($data_uint32);

// Test additional INT64 values
$tensor_int64_extra = new ORT\Tensor\Persistent(
    "test_tensor_int64_extra",
    [2],
    [1000000000000, -1000000000000],
    ORT\Tensor::INT64
);
$data_int64_extra = $tensor_int64_extra->getData();
var_dump($data_int64_extra);
?>
--EXPECT--
array(2) {
  [0]=>
  int(127)
  [1]=>
  int(-128)
}
array(2) {
  [0]=>
  int(32767)
  [1]=>
  int(-32768)
}
array(2) {
  [0]=>
  int(1000000000000)
  [1]=>
  int(-1000000000000)
}
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
  int(1000000000000)
  [1]=>
  int(-1000000000000)
}
