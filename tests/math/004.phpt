--TEST--
Check ONNX\Math functions - rounding operations
--EXTENSIONS--
ort
--FILE--
<?php
try {
    // Create test tensor with decimal values
    $tensor = new ONNX\Tensor\Persistent(
        "test_tensor",
        [2, 2],
        [[-2.7, -1.3], [1.6, 2.9]],
        ONNX\Tensor::FLOAT
    );
    
    // Test abs
    echo "=== ABSOLUTE VALUE ===\n";
    $result = ONNX\Math\abs($tensor);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test sign
    echo "=== SIGN ===\n";
    $result = ONNX\Math\sign($tensor);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test ceil
    echo "=== CEILING ===\n";
    $result = ONNX\Math\ceil($tensor);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test floor
    echo "=== FLOOR ===\n";
    $result = ONNX\Math\floor($tensor);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test round
    echo "=== ROUND ===\n";
    $result = ONNX\Math\round($tensor);
    echo "Result type: " . get_class($result) . "\n";
    
    echo "Rounding and utility tests passed!\n";
} catch (Error $e) {
    echo "Error: " . $e->getMessage() . "\n";
}
?>
--EXPECTF--
=== ABSOLUTE VALUE ===
Result type: ONNX\Tensor\Transient
=== SIGN ===
Result type: ONNX\Tensor\Transient
=== CEILING ===
Result type: ONNX\Tensor\Transient
=== FLOOR ===
Result type: ONNX\Tensor\Transient
=== ROUND ===
Result type: ONNX\Tensor\Transient
Rounding and utility tests passed!
