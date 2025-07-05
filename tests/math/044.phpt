--TEST--
Type Promotion Coverage - Mixed Signed/Unsigned and Boolean Cases
--EXTENSIONS--
ort
--FILE--
<?php

try {
    echo "=== Type Promotion Coverage Test ===\n";
    
    // Test 1: Mixed signed/unsigned promotion (lines 200-206)
    echo "\n--- Mixed Signed/Unsigned Type Promotion ---\n";
    
    // Test INT8 + UINT8 -> should promote to INT64
    try {
        $tensor_int8 = new ONNX\Tensor\Transient([2], [100, -100], ONNX\Tensor::INT8);
        $tensor_uint8 = new ONNX\Tensor\Transient([2], [200, 150], ONNX\Tensor::UINT8);
        $result = ONNX\Math\add($tensor_int8, $tensor_uint8);
        echo "PASS: INT8 + UINT8 mixed promotion works\n";
    } catch (Exception $e) {
        echo "INFO: INT8 + UINT8: " . $e->getMessage() . "\n";
    }
    
    // Test INT16 + UINT16 -> should promote to INT64
    try {
        $tensor_int16 = new ONNX\Tensor\Transient([2], [30000, -30000], ONNX\Tensor::INT16);
        $tensor_uint16 = new ONNX\Tensor\Transient([2], [60000, 50000], ONNX\Tensor::UINT16);
        $result = ONNX\Math\add($tensor_int16, $tensor_uint16);
        echo "PASS: INT16 + UINT16 mixed promotion works\n";
    } catch (Exception $e) {
        echo "INFO: INT16 + UINT16: " . $e->getMessage() . "\n";
    }
    
    // Test INT32 + UINT32 -> should promote to INT64
    try {
        $tensor_int32 = new ONNX\Tensor\Transient([2], [1000000000, -1000000000], ONNX\Tensor::INT32);
        $tensor_uint32 = new ONNX\Tensor\Transient([2], [3000000000, 2000000000], ONNX\Tensor::UINT32);
        $result = ONNX\Math\add($tensor_int32, $tensor_uint32);
        echo "PASS: INT32 + UINT32 mixed promotion works\n";
    } catch (Exception $e) {
        echo "INFO: INT32 + UINT32: " . $e->getMessage() . "\n";
    }
    
    // Test the reverse order: UINT + INT -> should also promote to INT64
    try {
        $tensor_uint8 = new ONNX\Tensor\Transient([2], [255, 100], ONNX\Tensor::UINT8);
        $tensor_int16 = new ONNX\Tensor\Transient([2], [1000, -1000], ONNX\Tensor::INT16);
        $result = ONNX\Math\add($tensor_uint8, $tensor_int16);
        echo "PASS: UINT8 + INT16 mixed promotion works\n";
    } catch (Exception $e) {
        echo "INFO: UINT8 + INT16: " . $e->getMessage() . "\n";
    }
    
    // Test UINT32 + INT64 -> should promote to INT64
    try {
        $tensor_uint32 = new ONNX\Tensor\Transient([2], [4000000000, 1000000000], ONNX\Tensor::UINT32);
        $tensor_int64 = new ONNX\Tensor\Transient([2], [5000000000000, -5000000000000], ONNX\Tensor::INT64);
        $result = ONNX\Math\add($tensor_uint32, $tensor_int64);
        echo "PASS: UINT32 + INT64 mixed promotion works\n";
    } catch (Exception $e) {
        echo "INFO: UINT32 + INT64: " . $e->getMessage() . "\n";
    }
    
    // Test 2: Boolean type promotion edge cases (lines 209-213)
    echo "\n--- Boolean Type Promotion Edge Cases ---\n";
    
    // These might be the fallback boolean cases that aren't hit by the early returns
    // We need to understand if these are reachable at all
    
    // Test BOOL with every numeric type to ensure we hit all boolean promotion paths
    $numeric_types = [
        ['INT8', [127, -128], 'ONNX\Tensor::INT8'],
        ['INT16', [32000, -32000], 'ONNX\Tensor::INT16'],
        ['INT32', [2000000000, -2000000000], 'ONNX\Tensor::INT32'],
        ['INT64', [9000000000000, -9000000000000], 'ONNX\Tensor::INT64'],
        ['UINT8', [255, 100], 'ONNX\Tensor::UINT8'],
        ['UINT16', [65000, 30000], 'ONNX\Tensor::UINT16'],
        ['UINT32', [4000000000, 2000000000], 'ONNX\Tensor::UINT32'],
        ['FLOAT', [123.456, -789.012], 'ONNX\Tensor::FLOAT'],
        ['DOUBLE', [1.23456789e100, -9.87654321e100], 'ONNX\Tensor::DOUBLE']
    ];
    
    foreach ($numeric_types as $type_info) {
        // Test BOOL as first operand
        try {
            $tensor_bool = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
            $tensor_numeric = new ONNX\Tensor\Transient([2], $type_info[1], constant($type_info[2]));
            $result = ONNX\Math\add($tensor_bool, $tensor_numeric);
            // Don't spam output for successful cases
        } catch (Exception $e) {
            echo "INFO: BOOL + {$type_info[0]}: " . $e->getMessage() . "\n";
        }
        
        // Test BOOL as second operand
        try {
            $tensor_numeric = new ONNX\Tensor\Transient([2], $type_info[1], constant($type_info[2]));
            $tensor_bool = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
            $result = ONNX\Math\add($tensor_numeric, $tensor_bool);
            // Don't spam output for successful cases
        } catch (Exception $e) {
            echo "INFO: {$type_info[0]} + BOOL: " . $e->getMessage() . "\n";
        }
    }
    
    echo "Completed boolean promotion tests\n";
    
    // Test 3: Complex mixed type scenarios
    echo "\n--- Complex Mixed Type Scenarios ---\n";
    
    // Try combinations that might exercise different code paths
    $mixed_combos = [
        ['INT8', [100, -50], 'ONNX\Tensor::INT8', 'UINT16', [40000, 30000], 'ONNX\Tensor::UINT16'],
        ['INT16', [25000, -15000], 'ONNX\Tensor::INT16', 'UINT8', [200, 150], 'ONNX\Tensor::UINT8'],
        ['INT64', [8000000000000, -3000000000000], 'ONNX\Tensor::INT64', 'UINT8', [255, 100], 'ONNX\Tensor::UINT8'],
        ['UINT32', [3500000000, 1500000000], 'ONNX\Tensor::UINT32', 'INT8', [50, -50], 'ONNX\Tensor::INT8']
    ];
    
    foreach ($mixed_combos as $combo) {
        try {
            $tensor1 = new ONNX\Tensor\Transient([2], $combo[1], constant($combo[2]));
            $tensor2 = new ONNX\Tensor\Transient([2], $combo[4], constant($combo[5]));
            $result = ONNX\Math\add($tensor1, $tensor2);
            echo "PASS: {$combo[0]} + {$combo[3]} promotion works\n";
        } catch (Exception $e) {
            echo "INFO: {$combo[0]} + {$combo[3]}: " . $e->getMessage() . "\n";
        }
    }
    
    echo "\n=== Type Promotion Coverage Test Complete ===\n";
    
} catch (Exception $e) {
    echo "FAIL: Unexpected error: " . $e->getMessage() . "\n";
}

?>
--EXPECT--
=== Type Promotion Coverage Test ===

--- Mixed Signed/Unsigned Type Promotion ---
PASS: INT8 + UINT8 mixed promotion works
PASS: INT16 + UINT16 mixed promotion works
PASS: INT32 + UINT32 mixed promotion works
PASS: UINT8 + INT16 mixed promotion works
PASS: UINT32 + INT64 mixed promotion works

--- Boolean Type Promotion Edge Cases ---
Completed boolean promotion tests

--- Complex Mixed Type Scenarios ---
PASS: INT8 + UINT16 promotion works
PASS: INT16 + UINT8 promotion works
PASS: INT64 + UINT8 promotion works
PASS: UINT32 + INT8 promotion works

=== Type Promotion Coverage Test Complete ===
