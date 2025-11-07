--TEST--
Check ORT\Math functions - exponential and logarithmic operations
--EXTENSIONS--
ort
--FILE--
<?php
try {
    // Create test tensors
    $tensor1 = new ORT\Tensor\Persistent(
        "test_tensor1",
        [2, 2],
        [[1.0, 2.0], [3.0, 4.0]],
        ORT\Tensor::FLOAT32
    );
    
    $tensor2 = new ORT\Tensor\Persistent(
        "test_tensor2",
        [2, 2], 
        [[8.0, 27.0], [64.0, 125.0]],
        ORT\Tensor::FLOAT32
    );
    
    // Test sqrt
    echo "=== SQUARE ROOT ===\n";
    $result = ORT\Math\sqrt($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test cbrt
    echo "=== CUBE ROOT ===\n";
    $result = ORT\Math\cbrt($tensor2);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test exp
    echo "=== EXPONENTIAL ===\n";
    $result = ORT\Math\exp($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test log
    echo "=== NATURAL LOGARITHM ===\n";
    $result = ORT\Math\log($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test exp2
    echo "=== EXPONENTIAL BASE 2 ===\n";
    $result = ORT\Math\exp2($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test log2
    echo "=== LOGARITHM BASE 2 ===\n";
    $result = ORT\Math\log2($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    // Test log10
    echo "=== LOGARITHM BASE 10 ===\n";
    $result = ORT\Math\log10($tensor1);
    echo "Result type: " . get_class($result) . "\n";
    
    echo "Exponential and logarithmic tests passed!\n";
} catch (Error $e) {
    echo "Error: " . $e->getMessage() . "\n";
}
?>
--EXPECTF--
=== SQUARE ROOT ===
Result type: ORT\Tensor\Transient
=== CUBE ROOT ===
Result type: ORT\Tensor\Transient
=== EXPONENTIAL ===
Result type: ORT\Tensor\Transient
=== NATURAL LOGARITHM ===
Result type: ORT\Tensor\Transient
=== EXPONENTIAL BASE 2 ===
Result type: ORT\Tensor\Transient
=== LOGARITHM BASE 2 ===
Result type: ORT\Tensor\Transient
=== LOGARITHM BASE 10 ===
Result type: ORT\Tensor\Transient
Exponential and logarithmic tests passed!
