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
        $tensor_bool = new ORT\Tensor\Transient([2], [true, false], ORT\Tensor::BOOL);
        $tensor_int8 = new ORT\Tensor\Transient([2], [100, -100], ORT\Tensor::INT8);
        $result = ORT\Math\add($tensor_bool, $tensor_int8);
        echo "PASS: BOOL + INT8 casting works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + INT8: " . $e->getMessage() . "\n";
    }
    
    // Test 2: BOOL to INT16 conversion (line 80)
    echo "\n--- Testing BOOL to INT16 ---\n";
    try {
        $tensor_bool = new ORT\Tensor\Transient([2], [true, false], ORT\Tensor::BOOL);
        $tensor_int16 = new ORT\Tensor\Transient([2], [30000, -30000], ORT\Tensor::INT16);
        $result = ORT\Math\add($tensor_bool, $tensor_int16);
        echo "PASS: BOOL + INT16 casting works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + INT16: " . $e->getMessage() . "\n";
    }
    
    // Test 3: BOOL to UINT8 conversion (line 83)
    echo "\n--- Testing BOOL to UINT8 ---\n";
    try {
        $tensor_bool = new ORT\Tensor\Transient([2], [true, false], ORT\Tensor::BOOL);
        $tensor_uint8 = new ORT\Tensor\Transient([2], [200, 100], ORT\Tensor::UINT8);
        $result = ORT\Math\add($tensor_bool, $tensor_uint8);
        echo "PASS: BOOL + UINT8 casting works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + UINT8: " . $e->getMessage() . "\n";
    }
    
    // Test 4: BOOL to BOOL conversion (line 87) - same type, should hit fast path
    echo "\n--- Testing BOOL to BOOL (same type) ---\n";
    try {
        $tensor_bool1 = new ORT\Tensor\Transient([2], [true, false], ORT\Tensor::BOOL);
        $tensor_bool2 = new ORT\Tensor\Transient([2], [false, true], ORT\Tensor::BOOL);
        $result = ORT\Math\add($tensor_bool1, $tensor_bool2);
        echo "PASS: BOOL + BOOL same type works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + BOOL: " . $e->getMessage() . "\n";
    }
    
    // Test 5: Try to hit FLOAT64 macro default case (line 64)
    // This is tricky - we need FLOAT64 source with an unsupported destination
    // Let's try combinations that might hit default branches
    echo "\n--- Testing FLOAT64 with Comprehensive Destinations ---\n";
    
    // FLOAT64 to INT8
    try {
        $tensor_FLOAT64 = new ORT\Tensor\Transient([2], [123.456, -789.012], ORT\Tensor::FLOAT64);
        $tensor_int8 = new ORT\Tensor\Transient([2], [50, -50], ORT\Tensor::INT8);
        $result = ORT\Math\add($tensor_FLOAT64, $tensor_int8);
        echo "PASS: FLOAT64 + INT8 casting works\n";
    } catch (Exception $e) {
        echo "INFO: FLOAT64 + INT8: " . $e->getMessage() . "\n";
    }
    
    // FLOAT64 to INT16
    try {
        $tensor_FLOAT64 = new ORT\Tensor\Transient([2], [12345.67, -23456.78], ORT\Tensor::FLOAT64);
        $tensor_int16 = new ORT\Tensor\Transient([2], [1000, -1000], ORT\Tensor::INT16);
        $result = ORT\Math\add($tensor_FLOAT64, $tensor_int16);
        echo "PASS: FLOAT64 + INT16 casting works\n";
    } catch (Exception $e) {
        echo "INFO: FLOAT64 + INT16: " . $e->getMessage() . "\n";
    }
    
    // FLOAT64 to INT32
    try {
        $tensor_FLOAT64 = new ORT\Tensor\Transient([2], [1234567.89, -9876543.21], ORT\Tensor::FLOAT64);
        $tensor_int32 = new ORT\Tensor\Transient([2], [100000, -100000], ORT\Tensor::INT32);
        $result = ORT\Math\add($tensor_FLOAT64, $tensor_int32);
        echo "PASS: FLOAT64 + INT32 casting works\n";
    } catch (Exception $e) {
        echo "INFO: FLOAT64 + INT32: " . $e->getMessage() . "\n";
    }
    
    // FLOAT64 to UINT8
    try {
        $tensor_FLOAT64 = new ORT\Tensor\Transient([2], [100.5, 200.7], ORT\Tensor::FLOAT64);
        $tensor_uint8 = new ORT\Tensor\Transient([2], [10, 20], ORT\Tensor::UINT8);
        $result = ORT\Math\add($tensor_FLOAT64, $tensor_uint8);
        echo "PASS: FLOAT64 + UINT8 casting works\n";
    } catch (Exception $e) {
        echo "INFO: FLOAT64 + UINT8: " . $e->getMessage() . "\n";
    }
    
    // Test 6: Try to hit the unsupported type path (line 55)
    // This is very difficult since we validate types before calling cast_element
    // But let's try edge cases that might trigger it
    echo "\n--- Testing Edge Cases for Unsupported Type Path ---\n";
    
    // Create tensors with extreme values that might trigger edge cases
    try {
        $tensor_float_inf = new ORT\Tensor\Transient([2], [INF, -INF], ORT\Tensor::FLOAT32);
        $tensor_FLOAT64_nan = new ORT\Tensor\Transient([2], [NAN, 1.0], ORT\Tensor::FLOAT64);
        $result = ORT\Math\add($tensor_float_inf, $tensor_FLOAT64_nan);
        echo "PASS: Infinity/NaN handling works\n";
    } catch (Exception $e) {
        echo "INFO: INF/NAN test: " . $e->getMessage() . "\n";
    }
    
    // Test 7: Comprehensive type matrix to hit any remaining paths
    echo "\n--- Testing Comprehensive Type Matrix ---\n";
    
    $types_to_test = [
        ['FLOAT32', [1.5, 2.5]],
        ['FLOAT64', [1.123456789, 2.987654321]],
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
                    $tensor1 = new ORT\Tensor\Transient([2], $type1[1], constant('ORT\Tensor::' . $type1[0]));
                    $tensor2 = new ORT\Tensor\Transient([2], $type2[1], constant('ORT\Tensor::' . $type2[0]));
                    $result = ORT\Math\add($tensor1, $tensor2);
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

--- Testing FLOAT64 with Comprehensive Destinations ---
PASS: FLOAT64 + INT8 casting works
PASS: FLOAT64 + INT16 casting works
PASS: FLOAT64 + INT32 casting works
PASS: FLOAT64 + UINT8 casting works

--- Testing Edge Cases for Unsupported Type Path ---
PASS: Infinity/NaN handling works

--- Testing Comprehensive Type Matrix ---
PASS: FLOAT32 + FLOAT64 works
PASS: FLOAT32 + INT8 works
PASS: FLOAT32 + INT16 works
Tested 72 type combinations

=== Maximum Coverage Test Complete ===
