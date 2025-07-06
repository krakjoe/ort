--TEST--
Check ONNX\Math functions - pow and advanced operations
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
    
    // Test pow with tensor
    echo "=== pow (TENSOR) ===\n";
    $result = ONNX\Math\pow($tensor1, $tensor2);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test pow with scalar
    echo "=== pow (SCALAR) ===\n";
    $result = ONNX\Math\pow($tensor1, 2.0);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test matrix multiplication
    echo "=== MATRIX MULTIPLICATION ===\n";
    $result = ONNX\Math\matmul($tensor1, $tensor2);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test sum
    echo "=== SUM ===\n";
    $result = ONNX\Math\sum($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    echo "pow and advanced operation tests passed!\n";
} catch (Error $e) {
    echo "Error: " . $e->getMessage() . "\n";
}
?>
--EXPECTF--
=== pow (TENSOR) ===
Result type: ONNX\Tensor\Transient
=== pow (SCALAR) ===
Result type: ONNX\Tensor\Transient
=== MATRIX MULTIPLICATION ===
Result type: ONNX\Tensor\Transient
=== SUM ===
Result type: ONNX\Tensor\Transient
pow and advanced operation tests passed!
