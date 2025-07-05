--TEST--
Double sum with axis coverage test
--EXTENSIONS--
ort
--FILE--
<?php
// Test DOUBLE sum with axis
$tensor = new ONNX\Tensor\Transient([2, 3], [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], ONNX\Tensor::DOUBLE);
$result = ONNX\Math\sum($tensor, 0);
echo "DOUBLE axis 0: " . implode(',', $result->getData()) . "\n";

$result = ONNX\Math\sum($tensor, 1);
echo "DOUBLE axis 1: " . implode(',', $result->getData()) . "\n";
?>
--EXPECT--
DOUBLE axis 0: 6,8,10
DOUBLE axis 1: 7.5,16.5
