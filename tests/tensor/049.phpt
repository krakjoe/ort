--TEST--
Check Tensor getData method edge cases and data type coverage
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
// Test with 1D tensor to cover different depth scenarios
$tensor_1d = new ORT\Tensor\Persistent(
    "tensor_1d",
    [5],
    [1, 2, 3, 4, 5],
    ORT\Tensor::INT32
);

// Test getData on 1D tensor with depth 0 (should work)
$data_1d = $tensor_1d->getData(0, 0);
var_dump(count($data_1d));

// Test with edge case: small tensor with different starting depths
$tensor_small = new ORT\Tensor\Persistent(
    "tensor_small",
    [2, 2],
    [[1, 2], [3, 4]],
    ORT\Tensor::INT32
);

// Test getData starting from depth 1 (should extract sub-arrays)
$data_depth1 = $tensor_small->getData(0, 1);
var_dump($data_depth1);

// Test BOOL type with getData to ensure BOOL branch is covered
$tensor_bool = new ORT\Tensor\Persistent(
    "tensor_bool",
    [2, 2],
    [[true, false], [false, true]],
    ORT\Tensor::BOOL
);

$data_bool = $tensor_bool->getData();
var_dump($data_bool);
?>
--EXPECT--
int(5)
array(2) {
  [0]=>
  int(1)
  [1]=>
  int(2)
}
array(2) {
  [0]=>
  array(2) {
    [0]=>
    bool(true)
    [1]=>
    bool(false)
  }
  [1]=>
  array(2) {
    [0]=>
    bool(false)
    [1]=>
    bool(true)
  }
}
