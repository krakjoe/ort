--TEST--
Check ONNX\Math functions - power and advanced operations
--EXTENSIONS--
ort
--FILE--
<?php
try {
    // Create test tensors
    $tensor1 = new ONNX\Tensor\Persistent(
        "base_tensor",
        [2, 2],
        [[2.0, 3.0], [4.0, 5.0]],
        ONNX\Tensor::FLOAT
    );
    
    $tensor2 = new ONNX\Tensor\Persistent(
        "exponent_tensor",
        [2, 2],
        [[2.0, 3.0], [2.0, 2.0]],
        ONNX\Tensor::FLOAT
    );
    
    // Test power with tensor
    echo "=== POWER (TENSOR) ===\n";
    $result = ONNX\Math\power($tensor1, $tensor2);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test power with scalar
    echo "=== POWER (SCALAR) ===\n";
    $result = ONNX\Math\power($tensor1, 2.0);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test matrix multiplication
    echo "=== MATRIX MULTIPLICATION ===\n";
    $result = ONNX\Math\matmul($tensor1, $tensor2);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test sum
    echo "=== SUM ===\n";
    $result = ONNX\Math\sum($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    echo "Power and advanced operation tests passed!\n";
} catch (Error $e) {
    echo "Error: " . $e->getMessage() . "\n";
}
?>
--EXPECTF--
=== POWER (TENSOR) ===
Result type: ONNX\Tensor\Transient
=== POWER (SCALAR) ===
Result type: ONNX\Tensor\Transient
=== MATRIX MULTIPLICATION ===
Result type: ONNX\Tensor\Transient
=== SUM ===
Result type: ONNX\Tensor\Transient
Power and advanced operation tests passed!
