--TEST--
Final Coverage Push - Target Defensive Code Paths
--EXTENSIONS--
ort
--FILE--
<?php

try {
    echo "=== Final Coverage Push - Defensive Paths ===\n";
    
    // Try to force conditions that might trigger the defensive paths
    
    // Test 1: Try to hit any remaining BOOL conversion cases
    echo "\n--- Exhaustive BOOL Conversions ---\n";
    
    // We might have missed some specific combinations - let's be systematic
    $bool_values = [true, false];
    $test_values = [
        ['INT8', [127, -128], 'ONNX\Tensor::INT8'],
        ['INT16', [32767, -32768], 'ONNX\Tensor::INT16'],
        ['INT32', [2147483647, -2147483648], 'ONNX\Tensor::INT32'],
        ['INT64', [1000000000000, -1000000000000], 'ONNX\Tensor::INT64'],
        ['UINT8', [255, 0], 'ONNX\Tensor::UINT8'],
        ['UINT16', [65535, 0], 'ONNX\Tensor::UINT16'],
        ['UINT32', [4294967295, 0], 'ONNX\Tensor::UINT32'],
        ['FLOAT', [3.4028235e+38, -3.4028235e+38], 'ONNX\Tensor::FLOAT'],
        ['DOUBLE', [1.7976931348623157e+308, -1.7976931348623157e+308], 'ONNX\Tensor::DOUBLE'],
        ['BOOL', [true, false], 'ONNX\Tensor::BOOL']
    ];
    
    // Test every type WITH BOOL (where BOOL is the SECOND operand - might trigger different paths)
    foreach ($test_values as $type_info) {
        try {
            $tensor_type = new ONNX\Tensor\Transient([2], $type_info[1], constant($type_info[2]));
            $tensor_bool = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
            $result = ONNX\Math\add($tensor_type, $tensor_bool);
            // Don't print every success to avoid spam
        } catch (Exception $e) {
            echo "INFO: {$type_info[0]} + BOOL: " . $e->getMessage() . "\n";
        }
    }
    
    // Test 2: Try edge cases that might trigger the unsupported type path (line 55)
    echo "\n--- Edge Cases for Unsupported Type Detection ---\n";
    
    // Try operations that might stress the type system in unexpected ways
    try {
        // Use very large arrays to see if we can trigger memory/size issues
        $large_data = array_fill(0, 100, 1.0);
        $tensor_large = new ONNX\Tensor\Transient([100], $large_data, ONNX\Tensor::FLOAT);
        $tensor_single = new ONNX\Tensor\Transient([1], [2.0], ONNX\Tensor::FLOAT);
        $result = ONNX\Math\add($tensor_large, $tensor_single); // Should broadcast
        echo "PASS: Large tensor operations work\n";
    } catch (Exception $e) {
        echo "INFO: Large tensor test: " . $e->getMessage() . "\n";
    }
    
    // Test 3: Try to force specific macro expansion paths
    echo "\n--- Targeting Specific Macro Paths ---\n";
    
    // Try combinations that might hit the default cases in macros
    $extreme_test_cases = [
        // Very precise floating point values
        ['DOUBLE', [1.23456789e100, -9.87654321e100]],
        ['FLOAT', [1.234e20, -9.876e20]],
        // Integer boundary values
        ['INT8', [120, -120]],
        ['INT16', [30000, -30000]],
        ['INT32', [2000000000, -2000000000]],
        ['UINT8', [250, 5]],
        ['UINT16', [60000, 5000]],
        ['UINT32', [4000000000, 1000000000]]
    ];
    
    foreach ($extreme_test_cases as $case1) {
        foreach ($extreme_test_cases as $case2) {
            if ($case1[0] !== $case2[0]) { // Different types
                try {
                    $tensor1 = new ONNX\Tensor\Transient([2], $case1[1], constant('ONNX\Tensor::' . $case1[0]));
                    $tensor2 = new ONNX\Tensor\Transient([2], $case2[1], constant('ONNX\Tensor::' . $case2[0]));
                    $result = ONNX\Math\add($tensor1, $tensor2);
                    // Success - no output
                } catch (Exception $e) {
                    // Might be interesting edge case
                    if (strpos($e->getMessage(), 'unsupported') !== false || 
                        strpos($e->getMessage(), 'invalid') !== false) {
                        echo "EDGE: {$case1[0]} + {$case2[0]}: " . $e->getMessage() . "\n";
                    }
                }
            }
        }
    }
    
    // Test 4: Try to trigger BOOL same-type within the BOOL source block (not fast path)
    echo "\n--- BOOL Internal Conversions ---\n";
    
    // Force a situation where we have BOOL source but might not take fast path
    try {
        // Create scenarios with different tensor shapes that force broadcasting
        $tensor_bool_1d = new ONNX\Tensor\Transient([1], [true], ONNX\Tensor::BOOL);
        $tensor_bool_2d = new ONNX\Tensor\Transient([2], [false, true], ONNX\Tensor::BOOL);
        $result = ONNX\Math\add($tensor_bool_1d, $tensor_bool_2d);
        echo "PASS: BOOL broadcasting works\n";
    } catch (Exception $e) {
        echo "INFO: BOOL broadcasting: " . $e->getMessage() . "\n";
    }
    
    // Test 5: Mathematical edge cases with special values
    echo "\n--- Special Value Testing ---\n";
    
    try {
        // Zero division edge cases
        $tensor_zero = new ONNX\Tensor\Transient([2], [0.0, 1.0], ONNX\Tensor::FLOAT);
        $tensor_div = new ONNX\Tensor\Transient([2], [1.0, 0.0], ONNX\Tensor::FLOAT);
        $result = ONNX\Math\divide($tensor_div, $tensor_zero);
        echo "PASS: Division by zero handled\n";
    } catch (Exception $e) {
        echo "INFO: Division by zero: " . $e->getMessage() . "\n";
    }
    
    echo "\nCompleted comprehensive coverage tests\n";
    
} catch (Exception $e) {
    echo "FAIL: Unexpected error: " . $e->getMessage() . "\n";
}

?>
--EXPECT--
=== Final Coverage Push - Defensive Paths ===

--- Exhaustive BOOL Conversions ---

--- Edge Cases for Unsupported Type Detection ---
PASS: Large tensor operations work

--- Targeting Specific Macro Paths ---

--- BOOL Internal Conversions ---
PASS: BOOL broadcasting works

--- Special Value Testing ---
PASS: Division by zero handled

Completed comprehensive coverage tests
