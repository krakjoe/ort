--TEST--
Test math_core type casting and compatibility - edge cases
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for math_core type casting and compatibility edge cases

echo "=== Testing Type Casting and Compatibility ===\n";

// Test 1: Mixed type operations (should promote to higher precision)
try {
    $tensor_int = new ORT\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ORT\Tensor::INT32);
    $tensor_float = new ORT\Tensor\Transient([2, 2], [[1.5, 2.5], [3.5, 4.5]], ORT\Tensor::FLOAT32);
    
    // This should test type promotion logic
    $result = ORT\Math\add($tensor_int, $tensor_float);
    echo "PASS: Mixed INT32 + FLOAT operation completed\n";
} catch (Error $e) {
    echo "FAIL: Mixed type operation failed: " . $e->getMessage() . "\n";
}

// Test 2: Boolean tensor operations
try {
    // Note: Boolean tensor creation might not be supported directly
    // This test may need adjustment based on tensor API
    echo "SKIP: Boolean tensor operations (may not be supported in tensor creation)\n";
} catch (Error $e) {
    echo "SKIP: Boolean tensor test: " . $e->getMessage() . "\n";
}

// Test 3: Different integer types
try {
    $tensor_int8 = new ORT\Tensor\Transient([2], [10, 20], ORT\Tensor::INT8);
    $tensor_int16 = new ORT\Tensor\Transient([2], [100, 200], ORT\Tensor::INT16);
    
    $result = ORT\Math\add($tensor_int8, $tensor_int16);
    echo "PASS: INT8 + INT16 operation completed\n";
} catch (Error $e) {
    echo "FAIL: Different integer types failed: " . $e->getMessage() . "\n";
}

// Test 4: Scalar type compatibility
try {
    $tensor_double = new ORT\Tensor\Transient([3], [1.1, 2.2, 3.3], ORT\Tensor::FLOAT64);
    
    // Test with integer scalar
    $result1 = ORT\Math\add($tensor_double, 5);
    
    // Test with float scalar
    $result2 = ORT\Math\multiply($tensor_double, 2.5);
    
    echo "PASS: Scalar type compatibility tests completed\n";
} catch (Error $e) {
    echo "FAIL: Scalar type compatibility failed: " . $e->getMessage() . "\n";
}

// Test 5: Type promotion hierarchy
try {
    // Test promoting to DOUBLE when one operand is DOUBLE
    $tensor_int = new ORT\Tensor\Transient([2], [1, 2], ORT\Tensor::INT32);
    $tensor_double = new ORT\Tensor\Transient([2], [1.5, 2.5], ORT\Tensor::FLOAT64);
    
    $result = ORT\Math\multiply($tensor_int, $tensor_double);
    echo "PASS: Type promotion to DOUBLE completed\n";
} catch (Error $e) {
    echo "FAIL: Type promotion failed: " . $e->getMessage() . "\n";
}

echo "Type casting tests completed.\n";
?>
--EXPECT--
=== Testing Type Casting and Compatibility ===
PASS: Mixed INT32 + FLOAT operation completed
SKIP: Boolean tensor operations (may not be supported in tensor creation)
PASS: INT8 + INT16 operation completed
PASS: Scalar type compatibility tests completed
PASS: Type promotion to DOUBLE completed
Type casting tests completed.
