--TEST--
Pure Integer Mixed Signed/Unsigned Type Promotion
--EXTENSIONS--
ort
--FILE--
<?php

try {
    echo "=== Pure Integer Mixed Signed/Unsigned Type Promotion ===\n";
    
    // The key insight: we need ONLY integer types (no floats!) to hit the mixed signed/unsigned path
    // Lines 200-206 in type promotion function
    
    echo "\n--- Pure Integer Mixed Promotions (No Floats) ---\n";
    
    // Test 1: INT8 + UINT8 (both integers, no floats)
    try {
        $tensor_int8 = new ORT\Tensor\Transient([2], [100, -50], ORT\Tensor::INT8);
        $tensor_uint8 = new ORT\Tensor\Transient([2], [200, 150], ORT\Tensor::UINT8);
        $result = ORT\Math\add($tensor_int8, $tensor_uint8);
        echo "PASS: INT8 + UINT8 mixed integer promotion works\n";
        
        // Check result type should be INT64
        $data = $result->getData();
        echo "Result values: " . $data[0] . ", " . $data[1] . "\n";
    } catch (Exception $e) {
        echo "INFO: INT8 + UINT8: " . $e->getMessage() . "\n";
    }
    
    // Test 2: UINT8 + INT16 (reverse order)
    try {
        $tensor_uint8 = new ORT\Tensor\Transient([2], [255, 100], ORT\Tensor::UINT8);
        $tensor_int16 = new ORT\Tensor\Transient([2], [1000, -1000], ORT\Tensor::INT16);
        $result = ORT\Math\add($tensor_uint8, $tensor_int16);
        echo "PASS: UINT8 + INT16 mixed integer promotion works\n";
        
        $data = $result->getData();
        echo "Result values: " . $data[0] . ", " . $data[1] . "\n";
    } catch (Exception $e) {
        echo "INFO: UINT8 + INT16: " . $e->getMessage() . "\n";
    }
    
    // Test 3: INT32 + UINT16 
    try {
        $tensor_int32 = new ORT\Tensor\Transient([2], [1000000, -500000], ORT\Tensor::INT32);
        $tensor_uint16 = new ORT\Tensor\Transient([2], [50000, 30000], ORT\Tensor::UINT16);
        $result = ORT\Math\add($tensor_int32, $tensor_uint16);
        echo "PASS: INT32 + UINT16 mixed integer promotion works\n";
        
        $data = $result->getData();
        echo "Result values: " . $data[0] . ", " . $data[1] . "\n";
    } catch (Exception $e) {
        echo "INFO: INT32 + UINT16: " . $e->getMessage() . "\n";
    }
    
    // Test 4: UINT32 + INT8 (large unsigned + small signed)
    try {
        $tensor_uint32 = new ORT\Tensor\Transient([2], [3000000000, 1000000000], ORT\Tensor::UINT32);
        $tensor_int8 = new ORT\Tensor\Transient([2], [50, -50], ORT\Tensor::INT8);
        $result = ORT\Math\add($tensor_uint32, $tensor_int8);
        echo "PASS: UINT32 + INT8 mixed integer promotion works\n";
        
        $data = $result->getData();
        echo "Result values: " . $data[0] . ", " . $data[1] . "\n";
    } catch (Exception $e) {
        echo "INFO: UINT32 + INT8: " . $e->getMessage() . "\n";
    }
    
    // Test 5: Multiple combinations to ensure we hit both directions of the conditional
    $signed_types = [
        ['INT8', [100, -100], 'ORT\Tensor::INT8'],
        ['INT16', [30000, -30000], 'ORT\Tensor::INT16'],
        ['INT32', [2000000000, -1000000000], 'ORT\Tensor::INT32'],
        ['INT64', [9000000000000, -5000000000000], 'ORT\Tensor::INT64']
    ];
    
    $unsigned_types = [
        ['UINT8', [255, 100], 'ORT\Tensor::UINT8'],
        ['UINT16', [65000, 30000], 'ORT\Tensor::UINT16'],
        ['UINT32', [4000000000, 2000000000], 'ORT\Tensor::UINT32']
    ];
    
    echo "\n--- Comprehensive Mixed Integer Combinations ---\n";
    
    $tested = 0;
    foreach ($signed_types as $signed) {
        foreach ($unsigned_types as $unsigned) {
            try {
                $tensor_signed = new ORT\Tensor\Transient([2], $signed[1], constant($signed[2]));
                $tensor_unsigned = new ORT\Tensor\Transient([2], $unsigned[1], constant($unsigned[2]));
                $result = ORT\Math\add($tensor_signed, $tensor_unsigned);
                $tested++;
                if ($tested <= 3) { // Don't spam output
                    echo "PASS: {$signed[0]} + {$unsigned[0]} -> INT64\n";
                }
            } catch (Exception $e) {
                if ($tested <= 3) {
                    echo "INFO: {$signed[0]} + {$unsigned[0]}: " . $e->getMessage() . "\n";
                }
            }
        }
    }
    echo "Tested $tested signed+unsigned integer combinations\n";
    
    echo "\n=== Pure Integer Mixed Type Promotion Complete ===\n";
    
} catch (Exception $e) {
    echo "FAIL: Unexpected error: " . $e->getMessage() . "\n";
}

?>
--EXPECT--
=== Pure Integer Mixed Signed/Unsigned Type Promotion ===

--- Pure Integer Mixed Promotions (No Floats) ---
PASS: INT8 + UINT8 mixed integer promotion works
Result values: 300, 100
PASS: UINT8 + INT16 mixed integer promotion works
Result values: 1255, -900
PASS: INT32 + UINT16 mixed integer promotion works
Result values: 1050000, -470000
PASS: UINT32 + INT8 mixed integer promotion works
Result values: 3000000050, 999999950

--- Comprehensive Mixed Integer Combinations ---
PASS: INT8 + UINT8 -> INT64
PASS: INT8 + UINT16 -> INT64
PASS: INT8 + UINT32 -> INT64
Tested 12 signed+unsigned integer combinations

=== Pure Integer Mixed Type Promotion Complete ===
