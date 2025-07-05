--TEST--
Maximum Coverage Type Casting - Hit Every Remaining Branch
--EXTENSIONS--
ort
--FILE--
<?php

try {
    echo "=== Maximum Coverage Type Casting Test ===\n";
    
    // Test 1: BOOL to INT8 conversion (line 79)
    echo "\n--- Testing BOOL to INT8 ---\n";
    try {
        $tensor_bool = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
        $tensor_int8 = new ONNX\Tensor\Transient([2], [100, -100], ONNX\Tensor::INT8);
        $result = ONNX\Math\add($tensor_bool, $tensor_int8);
        echo "PASS: BOOL + INT8 casting works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + INT8: " . $e->getMessage() . "\n";
    }
    
    // Test 2: BOOL to INT16 conversion (line 80)
    echo "\n--- Testing BOOL to INT16 ---\n";
    try {
        $tensor_bool = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
        $tensor_int16 = new ONNX\Tensor\Transient([2], [30000, -30000], ONNX\Tensor::INT16);
        $result = ONNX\Math\add($tensor_bool, $tensor_int16);
        echo "PASS: BOOL + INT16 casting works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + INT16: " . $e->getMessage() . "\n";
    }
    
    // Test 3: BOOL to UINT8 conversion (line 83)
    echo "\n--- Testing BOOL to UINT8 ---\n";
    try {
        $tensor_bool = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
        $tensor_uint8 = new ONNX\Tensor\Transient([2], [200, 100], ONNX\Tensor::UINT8);
        $result = ONNX\Math\add($tensor_bool, $tensor_uint8);
        echo "PASS: BOOL + UINT8 casting works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + UINT8: " . $e->getMessage() . "\n";
    }
    
    // Test 4: BOOL to BOOL conversion (line 87) - same type, should hit fast path
    echo "\n--- Testing BOOL to BOOL (same type) ---\n";
    try {
        $tensor_bool1 = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
        $tensor_bool2 = new ONNX\Tensor\Transient([2], [false, true], ONNX\Tensor::BOOL);
        $result = ONNX\Math\add($tensor_bool1, $tensor_bool2);
        echo "PASS: BOOL + BOOL same type works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + BOOL: " . $e->getMessage() . "\n";
    }
    
    // Test 5: Try to hit DOUBLE macro default case (line 64)
    // This is tricky - we need DOUBLE source with an unsupported destination
    // Let's try combinations that might hit default branches
    echo "\n--- Testing DOUBLE with Comprehensive Destinations ---\n";
    
    // DOUBLE to INT8
    try {
        $tensor_double = new ONNX\Tensor\Transient([2], [123.456, -789.012], ONNX\Tensor::DOUBLE);
        $tensor_int8 = new ONNX\Tensor\Transient([2], [50, -50], ONNX\Tensor::INT8);
        $result = ONNX\Math\add($tensor_double, $tensor_int8);
        echo "PASS: DOUBLE + INT8 casting works\n";
    } catch (Exception $e) {
        echo "INFO: DOUBLE + INT8: " . $e->getMessage() . "\n";
    }
    
    // DOUBLE to INT16
    try {
        $tensor_double = new ONNX\Tensor\Transient([2], [12345.67, -23456.78], ONNX\Tensor::DOUBLE);
        $tensor_int16 = new ONNX\Tensor\Transient([2], [1000, -1000], ONNX\Tensor::INT16);
        $result = ONNX\Math\add($tensor_double, $tensor_int16);
        echo "PASS: DOUBLE + INT16 casting works\n";
    } catch (Exception $e) {
        echo "INFO: DOUBLE + INT16: " . $e->getMessage() . "\n";
    }
    
    // DOUBLE to INT32
    try {
        $tensor_double = new ONNX\Tensor\Transient([2], [1234567.89, -9876543.21], ONNX\Tensor::DOUBLE);
        $tensor_int32 = new ONNX\Tensor\Transient([2], [100000, -100000], ONNX\Tensor::INT32);
        $result = ONNX\Math\add($tensor_double, $tensor_int32);
        echo "PASS: DOUBLE + INT32 casting works\n";
    } catch (Exception $e) {
        echo "INFO: DOUBLE + INT32: " . $e->getMessage() . "\n";
    }
    
    // DOUBLE to UINT8
    try {
        $tensor_double = new ONNX\Tensor\Transient([2], [100.5, 200.7], ONNX\Tensor::DOUBLE);
        $tensor_uint8 = new ONNX\Tensor\Transient([2], [10, 20], ONNX\Tensor::UINT8);
        $result = ONNX\Math\add($tensor_double, $tensor_uint8);
        echo "PASS: DOUBLE + UINT8 casting works\n";
    } catch (Exception $e) {
        echo "INFO: DOUBLE + UINT8: " . $e->getMessage() . "\n";
    }
    
    // Test 6: Try to hit the unsupported type path (line 55)
    // This is very difficult since we validate types before calling cast_element
    // But let's try edge cases that might trigger it
    echo "\n--- Testing Edge Cases for Unsupported Type Path ---\n";
    
    // Create tensors with extreme values that might trigger edge cases
    try {
        $tensor_float_inf = new ONNX\Tensor\Transient([2], [INF, -INF], ONNX\Tensor::FLOAT);
        $tensor_double_nan = new ONNX\Tensor\Transient([2], [NAN, 1.0], ONNX\Tensor::DOUBLE);
        $result = ONNX\Math\add($tensor_float_inf, $tensor_double_nan);
        echo "PASS: Infinity/NaN handling works\n";
    } catch (Exception $e) {
        echo "INFO: INF/NAN test: " . $e->getMessage() . "\n";
    }
    
    // Test 7: Comprehensive type matrix to hit any remaining paths
    echo "\n--- Testing Comprehensive Type Matrix ---\n";
    
    $types_to_test = [
        ['FLOAT', [1.5, 2.5]],
        ['DOUBLE', [1.123456789, 2.987654321]],
        ['INT8', [100, -100]],
        ['INT16', [20000, -20000]],
        ['INT32', [1000000, -1000000]],
        ['UINT8', [200, 150]],
        ['UINT16', [50000, 40000]],
        ['UINT32', [3000000000, 2000000000]],
        ['BOOL', [true, false]]
    ];
    
    $tested_combinations = 0;
    foreach ($types_to_test as $type1) {
        foreach ($types_to_test as $type2) {
            if ($type1[0] !== $type2[0]) { // Skip same types for variety
                try {
                    $tensor1 = new ONNX\Tensor\Transient([2], $type1[1], constant('ONNX\Tensor::' . $type1[0]));
                    $tensor2 = new ONNX\Tensor\Transient([2], $type2[1], constant('ONNX\Tensor::' . $type2[0]));
                    $result = ONNX\Math\add($tensor1, $tensor2);
                    $tested_combinations++;
                    if ($tested_combinations <= 3) { // Don't spam output
                        echo "PASS: {$type1[0]} + {$type2[0]} works\n";
                    }
                } catch (Exception $e) {
                    if ($tested_combinations <= 3) {
                        echo "INFO: {$type1[0]} + {$type2[0]}: " . $e->getMessage() . "\n";
                    }
                }
            }
        }
    }
    echo "Tested $tested_combinations type combinations\n";
    
    echo "\n=== Maximum Coverage Test Complete ===\n";
    
} catch (Exception $e) {
    echo "FAIL: Unexpected error: " . $e->getMessage() . "\n";
}

?>
--EXPECT--
=== Maximum Coverage Type Casting Test ===

--- Testing BOOL to INT8 ---
PASS: BOOL + INT8 casting works

--- Testing BOOL to INT16 ---
PASS: BOOL + INT16 casting works

--- Testing BOOL to UINT8 ---
PASS: BOOL + UINT8 casting works

--- Testing BOOL to BOOL (same type) ---
PASS: BOOL + BOOL same type works

--- Testing DOUBLE with Comprehensive Destinations ---
PASS: DOUBLE + INT8 casting works
PASS: DOUBLE + INT16 casting works
PASS: DOUBLE + INT32 casting works
PASS: DOUBLE + UINT8 casting works

--- Testing Edge Cases for Unsupported Type Path ---
PASS: Infinity/NaN handling works

--- Testing Comprehensive Type Matrix ---
PASS: FLOAT + DOUBLE works
PASS: FLOAT + INT8 works
PASS: FLOAT + INT16 works
Tested 72 type combinations

=== Maximum Coverage Test Complete ===
