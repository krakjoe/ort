--TEST--
Check Tensor getData method with INT64 type and UINT64 unsupported error
--EXTENSIONS--
ort
--SKIPIF--
<?php
include (\sprintf("%s/../fixtures/skipif.php", dirname(__FILE__)));
?>
--FILE--
<?php
// Test INT64 type with getData to ensure INT64 branch is covered
$tensor_int64 = new ONNX\Tensor\Persistent(
    "tensor_int64",
    [2],
    [1000000000000, -1000000000000], // Large but valid int64 values
    \ONNX\Tensor::INT64
);

$data_int64 = $tensor_int64->getData();
var_dump($data_int64);

// Test that UINT64 type is not supported
try {
    // Use the numeric value directly since the constant no longer exists
    $tensor_uint64 = new ONNX\Tensor\Persistent(
        "tensor_uint64", 
        [2],
        [100, 0], 
        13 // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 value
    );
} catch (ONNX\Status\Tensor\InvalidType $e) {
    echo "UINT64 not supported: " . $e->getMessage() . "\n";
}
?>
--EXPECT--
array(2) {
  [0]=>
  int(1000000000000)
  [1]=>
  int(-1000000000000)
}
UINT64 not supported: UINT64 tensor type is not supported (values exceed PHP integer range)
