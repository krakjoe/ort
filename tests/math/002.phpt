--TEST--
Check ONNX\Math functions - trigonometric operations
--EXTENSIONS--
ort
--FILE--
<?php
try {
    // Create test tensor with values suitable for trig functions
    $tensor = new ONNX\Tensor\Persistent(
        "test_tensor",
        [2, 2],
        [[0.0, 1.570796], [3.141593, 4.712389]], // 0, π/2, π, 3π/2
        ONNX\Tensor::FLOAT
    );
    
    // Test sin
    echo "=== SINE ===\n";
    $result = ONNX\Math\sin($tensor);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test cos
    echo "=== COSINE ===\n";
    $result = ONNX\Math\cos($tensor);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test tan
    echo "=== TANGENT ===\n";
    $result = ONNX\Math\tan($tensor);
    echo "Result type: " . get_class($result) . "\n";
    
    echo "Trigonometric tests passed!\n";
} catch (Error $e) {
    echo "Error: " . $e->getMessage() . "\n";
}
?>
--EXPECTF--
=== SINE ===
Result type: ONNX\Tensor\Transient
=== COSINE ===
Result type: ONNX\Tensor\Transient
=== TANGENT ===
Result type: ONNX\Tensor\Transient
Trigonometric tests passed!
