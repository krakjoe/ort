--TEST--
Integer types sum with axis coverage test
--EXTENSIONS--
ort
--FILE--
<?php
// Test INT8 sum with axis
$tensor = new ONNX\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ONNX\Tensor::INT8);
$result = ONNX\Math\sum($tensor, 0);
echo "INT8 axis: " . implode(',', $result->getData()) . "\n";

// Test INT16 sum with axis
$tensor = new ONNX\Tensor\Transient([2, 2], [[100, 200], [300, 400]], ONNX\Tensor::INT16);
$result = ONNX\Math\sum($tensor, 0);
echo "INT16 axis: " . implode(',', $result->getData()) . "\n";

// Test INT32 sum with axis
$tensor = new ONNX\Tensor\Transient([2, 2], [[1000, 2000], [3000, 4000]], ONNX\Tensor::INT32);
$result = ONNX\Math\sum($tensor, 0);
echo "INT32 axis: " . implode(',', $result->getData()) . "\n";

// Test INT64 sum with axis
$tensor = new ONNX\Tensor\Transient([2, 2], [[10000, 20000], [30000, 40000]], ONNX\Tensor::INT64);
$result = ONNX\Math\sum($tensor, 0);
echo "INT64 axis: " . implode(',', $result->getData()) . "\n";

// Test UINT8 sum with axis
$tensor = new ONNX\Tensor\Transient([2, 2], [[10, 20], [30, 40]], ONNX\Tensor::UINT8);
$result = ONNX\Math\sum($tensor, 0);
echo "UINT8 axis: " . implode(',', $result->getData()) . "\n";

// Test UINT16 sum with axis
$tensor = new ONNX\Tensor\Transient([2, 2], [[100, 200], [300, 400]], ONNX\Tensor::UINT16);
$result = ONNX\Math\sum($tensor, 0);
echo "UINT16 axis: " . implode(',', $result->getData()) . "\n";

// Test UINT32 sum with axis
$tensor = new ONNX\Tensor\Transient([2, 2], [[1000, 2000], [3000, 4000]], ONNX\Tensor::UINT32);
$result = ONNX\Math\sum($tensor, 0);
echo "UINT32 axis: " . implode(',', $result->getData()) . "\n";

// Test BOOL sum with axis
$tensor = new ONNX\Tensor\Transient([2, 2], [[true, false], [false, true]], ONNX\Tensor::BOOL);
$result = ONNX\Math\sum($tensor, 0);
echo "BOOL axis: " . implode(',', array_map(function($x) { return $x ? 1 : 0; }, $result->getData())) . "\n";
?>
--EXPECT--
INT8 axis: 4,6
INT16 axis: 400,600
INT32 axis: 4000,6000
INT64 axis: 40000,60000
UINT8 axis: 40,60
UINT16 axis: 400,600
UINT32 axis: 4000,6000
BOOL axis: 1,1
