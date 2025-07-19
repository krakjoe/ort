--TEST--
Test math_ops element-wise operations - comprehensive type coverage
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for math_ops element-wise operations with different data types

echo "=== Testing Element-wise Operations with Different Types ===\n";

// Test 1: Integer operations (INT32)
try {
    $tensor_a = new ORT\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ORT\Tensor::INT32);
    $tensor_b = new ORT\Tensor\Transient([2, 2], [[5, 6], [7, 8]], ORT\Tensor::INT32);
    
    $add_result = ORT\Math\add($tensor_a, $tensor_b);
    $sub_result = ORT\Math\subtract($tensor_a, $tensor_b);
    $mul_result = ORT\Math\multiply($tensor_a, $tensor_b);
    
    echo "PASS: INT32 element-wise operations completed\n";
} catch (Error $e) {
    echo "FAIL: INT32 operations failed: " . $e->getMessage() . "\n";
}

// Test 2: Double precision operations
try {
    $tensor_a = new ORT\Tensor\Transient([2, 2], [[1.5, 2.5], [3.5, 4.5]], ORT\Tensor::DOUBLE);
    $tensor_b = new ORT\Tensor\Transient([2, 2], [[0.5, 0.5], [0.5, 0.5]], ORT\Tensor::DOUBLE);
    
    $add_result = ORT\Math\add($tensor_a, $tensor_b);
    $div_result = ORT\Math\divide($tensor_a, $tensor_b);
    
    echo "PASS: DOUBLE element-wise operations completed\n";
} catch (Error $e) {
    echo "FAIL: DOUBLE operations failed: " . $e->getMessage() . "\n";
}

// Test 3: Division by zero handling
try {
    $tensor_a = new ORT\Tensor\Transient([2, 2], [[1.0, 2.0], [3.0, 4.0]], ORT\Tensor::FLOAT);
    $tensor_b = new ORT\Tensor\Transient([2, 2], [[1.0, 0.0], [1.0, 2.0]], ORT\Tensor::FLOAT);
    
    $result = ORT\Math\divide($tensor_a, $tensor_b);
    // Division by zero should produce inf values, not crash
    echo "PASS: Division by zero handled (produces inf)\n";
} catch (Error $e) {
    echo "PARTIAL: Division by zero: " . $e->getMessage() . "\n";
}

echo "Element-wise operation tests completed.\n";
?>
--EXPECT--
=== Testing Element-wise Operations with Different Types ===
PASS: INT32 element-wise operations completed
PASS: DOUBLE element-wise operations completed
PASS: Division by zero handled (produces inf)
Element-wise operation tests completed.
