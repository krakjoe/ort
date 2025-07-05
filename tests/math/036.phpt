--TEST--
Math core utility functions coverage test
--EXTENSIONS--
ort
--FILE--
<?php
// Test operations that should exercise math_core utility functions

// Test different tensor shapes and types to exercise type promotion and validation
echo "Testing utility functions coverage...\n";

// Test broadcast compatibility by doing operations with different shaped tensors
try {
    $tensor1 = new ONNX\Tensor\Transient([2, 1], [[1.0], [2.0]], ONNX\Tensor::FLOAT);
    $tensor2 = new ONNX\Tensor\Transient([1, 3], [[3.0, 4.0, 5.0]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\add($tensor1, $tensor2);
    echo "Broadcast add succeeded\n";
} catch (Exception $e) {
    echo "Broadcast add failed: " . $e->getMessage() . "\n";
}

// Test type promotion by mixing different numeric types
try {
    $tensor1 = new ONNX\Tensor\Transient([2], [1, 2], ONNX\Tensor::INT32);
    $tensor2 = new ONNX\Tensor\Transient([2], [3.0, 4.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\add($tensor1, $tensor2);
    echo "Type promotion add succeeded\n";
} catch (Exception $e) {
    echo "Type promotion add failed: " . $e->getMessage() . "\n";
}

// Test incompatible types to exercise error paths
try {
    $tensor1 = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
    $tensor2 = new ONNX\Tensor\Transient([2], [3.0, 4.0], ONNX\Tensor::DOUBLE);
    $result = ONNX\Math\add($tensor1, $tensor2);
    echo "Bool-Double add succeeded\n";
} catch (Exception $e) {
    echo "Bool-Double add failed: " . $e->getMessage() . "\n";
}

echo "Math core utility test completed\n";
?>
--EXPECT--
Testing utility functions coverage...
Broadcast add succeeded
Type promotion add succeeded
Bool-Double add succeeded
Math core utility test completed
