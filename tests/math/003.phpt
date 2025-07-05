--TEST--
Check ONNX\Math functions - exponential and logarithmic operations
--EXTENSIONS--
ort
--FILE--
<?php
try {
    // Create test tensors
    $tensor1 = new ONNX\Tensor\Persistent(
        "test_tensor1",
        [2, 2],
        [[1.0, 2.0], [3.0, 4.0]],
        ONNX\Tensor::FLOAT
    );
    
    $tensor2 = new ONNX\Tensor\Persistent(
        "test_tensor2",
        [2, 2], 
        [[8.0, 27.0], [64.0, 125.0]],
        ONNX\Tensor::FLOAT
    );
    
    // Test sqrt
    echo "=== SQUARE ROOT ===\n";
    $result = ONNX\Math\sqrt($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test cbrt
    echo "=== CUBE ROOT ===\n";
    $result = ONNX\Math\cbrt($tensor2);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test exp
    echo "=== EXPONENTIAL ===\n";
    $result = ONNX\Math\exp($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test log
    echo "=== NATURAL LOGARITHM ===\n";
    $result = ONNX\Math\log($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test exp2
    echo "=== EXPONENTIAL BASE 2 ===\n";
    $result = ONNX\Math\exp2($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test log2
    echo "=== LOGARITHM BASE 2 ===\n";
    $result = ONNX\Math\log2($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test log10
    echo "=== LOGARITHM BASE 10 ===\n";
    $result = ONNX\Math\log10($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    echo "Exponential and logarithmic tests passed!\n";
} catch (Error $e) {
    echo "Error: " . $e->getMessage() . "\n";
}
?>
--EXPECTF--
=== SQUARE ROOT ===
Result type: ONNX\Tensor\Transient
=== CUBE ROOT ===
Result type: ONNX\Tensor\Transient
=== EXPONENTIAL ===
Result type: ONNX\Tensor\Transient
=== NATURAL LOGARITHM ===
Result type: ONNX\Tensor\Transient
=== EXPONENTIAL BASE 2 ===
Result type: ONNX\Tensor\Transient
=== LOGARITHM BASE 2 ===
Result type: ONNX\Tensor\Transient
=== LOGARITHM BASE 10 ===
Result type: ONNX\Tensor\Transient
Exponential and logarithmic tests passed!
