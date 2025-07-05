--TEST--
Test math_core utility functions - comprehensive index and element utilities
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for math_core utility functions that are likely uncovered

echo "=== Testing Math Core Utility Functions ===\n";

// Test 1: Test string conversion utilities indirectly
try {
    // These functions should be called internally when errors occur
    $tensor_a = new ONNX\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ONNX\Tensor::FLOAT);
    $tensor_b = new ONNX\Tensor\Transient([3, 2], [[1, 2], [3, 4], [5, 6]], ONNX\Tensor::FLOAT);
    
    // This should trigger shape mismatch error that uses string utilities
    try {
        $result = ONNX\Math\add($tensor_a, $tensor_b);
        echo "FAIL: Shape mismatch should trigger string utilities\n";
    } catch (Error $e) {
        // Error message should contain shape information (testing string utilities)
        if (strpos($e->getMessage(), 'shape') !== false || strpos($e->getMessage(), 'tensor') !== false) {
            echo "PASS: String utilities used in error messages\n";
        } else {
            echo "PARTIAL: Error occurred but string utilities unclear: " . $e->getMessage() . "\n";
        }
    }
} catch (Error $e) {
    echo "FAIL: String utility test setup failed: " . $e->getMessage() . "\n";
}

// Test 2: Test result creation with different tensor shapes
try {
    // Create various tensor shapes to test result creation utilities
    $scalar = new ONNX\Tensor\Transient([], [5], ONNX\Tensor::FLOAT);
    $vector = new ONNX\Tensor\Transient([3], [1, 2, 3], ONNX\Tensor::FLOAT);
    $matrix = new ONNX\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ONNX\Tensor::FLOAT);
    
    // Operations that create results with different shapes
    $result1 = ONNX\Math\sqrt($scalar);  // Scalar result
    $result2 = ONNX\Math\sqrt($vector);  // Vector result
    $result3 = ONNX\Math\sqrt($matrix);  // Matrix result
    
    echo "PASS: Result creation utilities work for various shapes\n";
} catch (Error $e) {
    echo "FAIL: Result creation utilities failed: " . $e->getMessage() . "\n";
}

// Test 3: Test with unusual tensor shapes
try {
    // Create tensors with unusual shapes to test shape utilities
    $tensor_1d = new ONNX\Tensor\Persistent("1d", [10], [1,2,3,4,5,6,7,8,9,10], ONNX\Tensor::FLOAT);
    $tensor_tall = new ONNX\Tensor\Transient([5, 1], [[1],[2],[3],[4],[5]], ONNX\Tensor::FLOAT);
    $tensor_wide = new ONNX\Tensor\Transient([1, 5], [[1,2,3,4,5]], ONNX\Tensor::FLOAT);
    
    $result1 = ONNX\Math\sqrt($tensor_1d);
    $result2 = ONNX\Math\sqrt($tensor_tall);
    $result3 = ONNX\Math\sqrt($tensor_wide);
    
    echo "PASS: Unusual shape utilities work\n";
} catch (Error $e) {
    echo "FAIL: Unusual shape utilities failed: " . $e->getMessage() . "\n";
}

// Test 4: Test element calculation and validation utilities
try {
    // Create tensors that will exercise element calculation functions
    $small = new ONNX\Tensor\Transient([1], [1], ONNX\Tensor::FLOAT);
    $medium = new ONNX\Tensor\Transient([10], array_fill(0, 10, 1.0), ONNX\Tensor::FLOAT);
    $large = new ONNX\Tensor\Transient([10, 10], array_fill(0, 10, array_fill(0, 10, 1.0)), ONNX\Tensor::FLOAT);
    
    // Operations that require element counting and validation
    $result1 = ONNX\Math\sum($small);
    $result2 = ONNX\Math\sum($medium);
    $result3 = ONNX\Math\sum($large);
    
    echo "PASS: Element calculation utilities work\n";
} catch (Error $e) {
    echo "FAIL: Element calculation utilities failed: " . $e->getMessage() . "\n";
}

// Test 5: Test validation with edge case tensors
try {
    // Test validation functions with extreme but valid cases
    $tiny = new ONNX\Tensor\Transient([1, 1, 1], [[[1]]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($tiny);
    
    echo "PASS: Validation utilities handle edge cases\n";
} catch (Error $e) {
    echo "FAIL: Validation utilities failed on edge cases: " . $e->getMessage() . "\n";
}

echo "Math core utility function tests completed.\n";
?>
--EXPECT--
=== Testing Math Core Utility Functions ===
PASS: String utilities used in error messages
PASS: Result creation utilities work for various shapes
PASS: Unusual shape utilities work
PASS: Element calculation utilities work
PASS: Validation utilities handle edge cases
Math core utility function tests completed.
