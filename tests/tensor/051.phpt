--TEST--
Check Tensor getData method with INT64 type
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
// Test INT64 type with getData to ensure INT64 branch is covered
$tensor_int64 = new ORT\Tensor\Persistent(
    "tensor_int64",
    [2],
    [1000000000000, -1000000000000], // Large but valid int64 values
    \ORT\Tensor::INT64
);

$data_int64 = $tensor_int64->getData();
var_dump($data_int64);
?>
--EXPECT--
array(2) {
  [0]=>
  int(1000000000000)
  [1]=>
  int(-1000000000000)
}