--TEST--
Integer types sum all elements coverage test
--EXTENSIONS--
ort
--FILE--
<?php
// Test INT8 sum all elements (no axis)
$tensor = new ONNX\Tensor\Transient([4], [1, 2, 3, 4], ONNX\Tensor::INT8);
$result = ONNX\Math\sum($tensor, null);
echo "INT8 sum all: " . $result->getData()[0] . "\n";

// Test INT16 sum all elements 
$tensor = new ONNX\Tensor\Transient([3], [100, 200, 300], ONNX\Tensor::INT16);
$result = ONNX\Math\sum($tensor, null);
echo "INT16 sum all: " . $result->getData()[0] . "\n";

// Test INT32 sum all elements
$tensor = new ONNX\Tensor\Transient([2], [1000, 2000], ONNX\Tensor::INT32);
$result = ONNX\Math\sum($tensor, null);
echo "INT32 sum all: " . $result->getData()[0] . "\n";

// Test INT64 sum all elements
$tensor = new ONNX\Tensor\Transient([2], [10000, 20000], ONNX\Tensor::INT64);
$result = ONNX\Math\sum($tensor, null);
echo "INT64 sum all: " . $result->getData()[0] . "\n";

// Test UINT8 sum all elements
$tensor = new ONNX\Tensor\Transient([3], [10, 20, 30], ONNX\Tensor::UINT8);
$result = ONNX\Math\sum($tensor, null);
echo "UINT8 sum all: " . $result->getData()[0] . "\n";

// Test UINT16 sum all elements
$tensor = new ONNX\Tensor\Transient([2], [1000, 2000], ONNX\Tensor::UINT16);
$result = ONNX\Math\sum($tensor, null);
echo "UINT16 sum all: " . $result->getData()[0] . "\n";

// Test UINT32 sum all elements
$tensor = new ONNX\Tensor\Transient([2], [50000, 60000], ONNX\Tensor::UINT32);
$result = ONNX\Math\sum($tensor, null);
echo "UINT32 sum all: " . $result->getData()[0] . "\n";

// Test BOOL sum all elements
$tensor = new ONNX\Tensor\Transient([4], [true, false, true, false], ONNX\Tensor::BOOL);
$result = ONNX\Math\sum($tensor, null);
echo "BOOL sum all: " . ($result->getData()[0] ? 1 : 0) . "\n";
?>
--EXPECT--
INT8 sum all: 10
INT16 sum all: 600
INT32 sum all: 3000
INT64 sum all: 30000
UINT8 sum all: 60
UINT16 sum all: 3000
UINT32 sum all: 110000
BOOL sum all: 1
