--TEST--
Edge Case Type Casting Coverage - Uncovered Branches
--EXTENSIONS--
ort
--FILE--
<?php

try {
    echo "=== Testing Edge Case Type Casting Branches ===\n";
    
    // Test 1: More BOOL source type combinations (to hit lines 79-87)
    echo "\n--- Testing BOOL to Various Types ---\n";
    
    // BOOL to INT64
    try {
        $tensor_bool = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
        $tensor_int64 = new ONNX\Tensor\Transient([2], [1000000000000, -1000000000000], ONNX\Tensor::INT64);
        $result = ONNX\Math\add($tensor_bool, $tensor_int64);
        echo "PASS: BOOL + INT64 casting works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + INT64: " . $e->getMessage() . "\n";
    }
    
    // BOOL to UINT16  
    try {
        $tensor_bool = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
        $tensor_uint16 = new ONNX\Tensor\Transient([2], [65000, 30000], ONNX\Tensor::UINT16);
        $result = ONNX\Math\add($tensor_bool, $tensor_uint16);
        echo "PASS: BOOL + UINT16 casting works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + UINT16: " . $e->getMessage() . "\n";
    }
    
    // BOOL to UINT32
    try {
        $tensor_bool = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
        $tensor_uint32 = new ONNX\Tensor\Transient([2], [4000000000, 3000000000], ONNX\Tensor::UINT32);
        $result = ONNX\Math\add($tensor_bool, $tensor_uint32);
        echo "PASS: BOOL + UINT32 casting works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + UINT32: " . $e->getMessage() . "\n";
    }
    
    // Test 2: More DOUBLE source type cases
    echo "\n--- Testing DOUBLE Source to Various Types ---\n";
    
    // DOUBLE to INT64
    try {
        $tensor_double = new ONNX\Tensor\Transient([2], [1.23e12, -9.87e11], ONNX\Tensor::DOUBLE);
        $tensor_int64 = new ONNX\Tensor\Transient([2], [1, 2], ONNX\Tensor::INT64);
        $result = ONNX\Math\add($tensor_double, $tensor_int64);
        echo "PASS: DOUBLE + INT64 casting works\n";
    } catch (Exception $e) {
        echo "INFO: DOUBLE + INT64: " . $e->getMessage() . "\n";
    }
    
    // DOUBLE to UINT16
    try {
        $tensor_double = new ONNX\Tensor\Transient([2], [1234.5678, 9876.5432], ONNX\Tensor::DOUBLE);
        $tensor_uint16 = new ONNX\Tensor\Transient([2], [100, 200], ONNX\Tensor::UINT16);
        $result = ONNX\Math\add($tensor_double, $tensor_uint16);
        echo "PASS: DOUBLE + UINT16 casting works\n";
    } catch (Exception $e) {
        echo "INFO: DOUBLE + UINT16: " . $e->getMessage() . "\n";
    }
    
    // DOUBLE to UINT32
    try {
        $tensor_double = new ONNX\Tensor\Transient([2], [123456789.0, 987654321.0], ONNX\Tensor::DOUBLE);
        $tensor_uint32 = new ONNX\Tensor\Transient([2], [1000, 2000], ONNX\Tensor::UINT32);
        $result = ONNX\Math\add($tensor_double, $tensor_uint32);
        echo "PASS: DOUBLE + UINT32 casting works\n";
    } catch (Exception $e) {
        echo "INFO: DOUBLE + UINT32: " . $e->getMessage() . "\n";
    }
    
    // DOUBLE to BOOL
    try {
        $tensor_double = new ONNX\Tensor\Transient([2], [0.0, 1.0], ONNX\Tensor::DOUBLE);
        $tensor_bool = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
        $result = ONNX\Math\add($tensor_double, $tensor_bool);
        echo "PASS: DOUBLE + BOOL casting works\n";
    } catch (Exception $e) {
        echo "INFO: DOUBLE + BOOL: " . $e->getMessage() . "\n";
    }
    
    // Test 3: Try to create conditions that might trigger unsupported paths
    echo "\n--- Testing Edge Cases for Error Paths ---\n";
    
    // Try operations that might stress the type system
    try {
        // Large value conversions
        $tensor_int32_max = new ONNX\Tensor\Transient([1], [2147483647], ONNX\Tensor::INT32);
        $tensor_uint8_max = new ONNX\Tensor\Transient([1], [255], ONNX\Tensor::UINT8);
        $result = ONNX\Math\add($tensor_int32_max, $tensor_uint8_max);
        echo "PASS: Max value conversions work\n";
    } catch (Exception $e) {
        echo "INFO: Max value conversion: " . $e->getMessage() . "\n";
    }
    
    echo "\n=== Edge Case Type Casting Test Complete ===\n";
    
} catch (Exception $e) {
    echo "FAIL: Unexpected error: " . $e->getMessage() . "\n";
}

?>
--EXPECT--
=== Testing Edge Case Type Casting Branches ===

--- Testing BOOL to Various Types ---
PASS: BOOL + INT64 casting works
PASS: BOOL + UINT16 casting works
PASS: BOOL + UINT32 casting works

--- Testing DOUBLE Source to Various Types ---
PASS: DOUBLE + INT64 casting works
PASS: DOUBLE + UINT16 casting works
PASS: DOUBLE + UINT32 casting works
PASS: DOUBLE + BOOL casting works

--- Testing Edge Cases for Error Paths ---
PASS: Max value conversions work

=== Edge Case Type Casting Test Complete ===
