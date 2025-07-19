--TEST--
Additional Type Casting Coverage - Less Common Type Combinations
--EXTENSIONS--
ort
--FILE--
<?php

try {
    echo "=== Testing Additional Type Casting Combinations ===\n";
    
    // Test 1: INT64 operations (less common in our current tests)
    echo "\n--- Testing INT64 Type Casting ---\n";
    try {
        $tensor_int64 = new ORT\Tensor\Transient([2], [9223372036854775800, -9223372036854775800], ORT\Tensor::INT64);
        $tensor_float = new ORT\Tensor\Transient([2], [1.0, 2.0], ORT\Tensor::FLOAT);
        $result = ORT\Math\add($tensor_int64, $tensor_float);
        echo "PASS: INT64 + FLOAT casting works\n";
    } catch (Exception $e) {
        echo "INFO: INT64 + FLOAT: " . $e->getMessage() . "\n";
    }
    
    // Test 2: DOUBLE source type casting (to exercise more cast paths)
    echo "\n--- Testing DOUBLE Source Type Casting ---\n";
    try {
        $tensor_double = new ORT\Tensor\Transient([2], [1.23456789012345, -2.98765432109876], ORT\Tensor::DOUBLE);
        $tensor_int8 = new ORT\Tensor\Transient([2], [10, 20], ORT\Tensor::INT8);
        $result = ORT\Math\add($tensor_double, $tensor_int8);
        echo "PASS: DOUBLE + INT8 casting works\n";
    } catch (Exception $e) {
        echo "INFO: DOUBLE + INT8: " . $e->getMessage() . "\n";
    }
    
    // Test 3: More BOOL conversions
    echo "\n--- Testing BOOL Source Type Casting ---\n";
    try {
        $tensor_bool = new ORT\Tensor\Transient([3], [true, false, true], ORT\Tensor::BOOL);
        $tensor_double = new ORT\Tensor\Transient([3], [1.5, 2.5, 3.5], ORT\Tensor::DOUBLE);
        $result = ORT\Math\add($tensor_bool, $tensor_double);
        echo "PASS: BOOL + DOUBLE casting works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + DOUBLE: " . $e->getMessage() . "\n";
    }
    
    // Test 4: INT16 type combinations
    echo "\n--- Testing INT16 Type Casting ---\n";
    try {
        $tensor_int16 = new ORT\Tensor\Transient([2], [32000, -32000], ORT\Tensor::INT16);
        $tensor_uint16 = new ORT\Tensor\Transient([2], [60000, 50000], ORT\Tensor::UINT16);
        $result = ORT\Math\add($tensor_int16, $tensor_uint16);
        echo "PASS: INT16 + UINT16 casting works\n";
    } catch (Exception $e) {
        echo "INFO: INT16 + UINT16: " . $e->getMessage() . "\n";
    }
    
    // Test 5: Edge case - same type operations (fast path)
    echo "\n--- Testing Same Type Fast Path ---\n";
    try {
        $tensor_a = new ORT\Tensor\Transient([2], [1.0, 2.0], ORT\Tensor::FLOAT);
        $tensor_b = new ORT\Tensor\Transient([2], [3.0, 4.0], ORT\Tensor::FLOAT);
        $result = ORT\Math\add($tensor_a, $tensor_b);
        echo "PASS: FLOAT + FLOAT same type works\n";
    } catch (Exception $e) {
        echo "INFO: FLOAT + FLOAT: " . $e->getMessage() . "\n";
    }
    
    // Test 6: Test unsupported type detection (this might be hard to trigger)
    echo "\n--- Testing Edge Cases ---\n";
    try {
        // Create a scenario that might trigger error paths
        $tensor_uint8 = new ORT\Tensor\Transient([1], [255], ORT\Tensor::UINT8);
        $tensor_int32 = new ORT\Tensor\Transient([1], [2147483647], ORT\Tensor::INT32);
        $result = ORT\Math\add($tensor_uint8, $tensor_int32);
        echo "PASS: UINT8 + INT32 edge values work\n";
    } catch (Exception $e) {
        echo "INFO: UINT8 + INT32 edge: " . $e->getMessage() . "\n";
    }
    
    echo "\n=== Type Casting Coverage Test Complete ===\n";
    
} catch (Exception $e) {
    echo "FAIL: Unexpected error: " . $e->getMessage() . "\n";
}

?>
--EXPECT--
=== Testing Additional Type Casting Combinations ===

--- Testing INT64 Type Casting ---
PASS: INT64 + FLOAT casting works

--- Testing DOUBLE Source Type Casting ---
PASS: DOUBLE + INT8 casting works

--- Testing BOOL Source Type Casting ---
PASS: BOOL + DOUBLE casting works

--- Testing INT16 Type Casting ---
PASS: INT16 + UINT16 casting works

--- Testing Same Type Fast Path ---
PASS: FLOAT + FLOAT same type works

--- Testing Edge Cases ---
PASS: UINT8 + INT32 edge values work

=== Type Casting Coverage Test Complete ===
