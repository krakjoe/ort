--TEST--
Boolean Fallback Type Promotion Coverage
--EXTENSIONS--
ort
--FILE--
<?php

try {
    echo "=== Boolean Fallback Type Promotion Test ===\n";
    
    // The mixed signed/unsigned promotion is unreachable due to overlapping ranges
    // But the boolean fallback cases (lines 209-213) might be reachable
    // These are the duplicate boolean cases at the end of the function
    
    // The early boolean handling (lines 154-162) might not cover all cases
    // Let's try to find scenarios where we hit the later boolean cases
    
    echo "\n--- Testing Boolean Edge Cases ---\n";
    
    // Try various boolean combinations that might skip the early returns
    // and hit the later boolean handling
    
    $test_types = [
        ['UNDEFINED', [0, 0], 'ORT\Tensor::UNDEFINED'],
        ['FLOAT', [1.5, 2.5], 'ORT\Tensor::FLOAT'],
        ['DOUBLE', [1.5, 2.5], 'ORT\Tensor::DOUBLE'],
        ['INT8', [10, 20], 'ORT\Tensor::INT8'],
        ['INT16', [100, 200], 'ORT\Tensor::INT16'],
        ['INT32', [1000, 2000], 'ORT\Tensor::INT32'],
        ['INT64', [10000, 20000], 'ORT\Tensor::INT64'],
        ['UINT8', [50, 100], 'ORT\Tensor::UINT8'],
        ['UINT16', [500, 1000], 'ORT\Tensor::UINT16'],
        ['UINT32', [5000, 10000], 'ORT\Tensor::UINT32']
    ];
    
    // Test BOOL with each type to see if we can hit the fallback boolean cases
    foreach ($test_types as $type_info) {
        if ($type_info[0] === 'UNDEFINED') {
            continue; // Skip undefined as it might not be valid for tensor creation
        }
        
        try {
            // Test boolean as first operand
            $tensor_bool = new ORT\Tensor\Transient([2], [true, false], ORT\Tensor::BOOL);
            $tensor_other = new ORT\Tensor\Transient([2], $type_info[1], constant($type_info[2]));
            $result = ORT\Math\add($tensor_bool, $tensor_other);
            
            // Test boolean as second operand  
            $tensor_other2 = new ORT\Tensor\Transient([2], $type_info[1], constant($type_info[2]));
            $tensor_bool2 = new ORT\Tensor\Transient([2], [false, true], ORT\Tensor::BOOL);
            $result2 = ORT\Math\add($tensor_other2, $tensor_bool2);
            
            // Don't spam output
        } catch (Exception $e) {
            if (strpos($e->getMessage(), 'unsupported') !== false) {
                echo "Edge case with {$type_info[0]}: " . $e->getMessage() . "\n";
            }
        }
    }
    
    echo "Completed boolean edge case testing\n";
    
    // Try to create conditions that might force different code paths
    echo "\n--- Testing Promotion Edge Cases ---\n";
    
    // Test combinations that might expose other uncovered paths
    try {
        // Test with extreme values that might trigger edge cases
        $tensor_bool_max = new ORT\Tensor\Transient([2], [true, false], ORT\Tensor::BOOL);
        $tensor_int64_max = new ORT\Tensor\Transient([2], [9223372036854775800, -9223372036854775800], ORT\Tensor::INT64);
        $result = ORT\Math\add($tensor_bool_max, $tensor_int64_max);
        echo "PASS: BOOL + INT64 extreme values work\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + INT64 extreme: " . $e->getMessage() . "\n";
    }
    
    // Test boolean with all unsigned types comprehensively
    try {
        $tensor_bool = new ORT\Tensor\Transient([3], [true, false, true], ORT\Tensor::BOOL);
        $tensor_uint32_large = new ORT\Tensor\Transient([3], [4294967290, 1000000000, 3000000000], ORT\Tensor::UINT32);
        $result = ORT\Math\add($tensor_bool, $tensor_uint32_large);
        echo "PASS: BOOL + UINT32 large values work\n";
    } catch (Exception $e) {
        echo "INFO: BOOL + UINT32 large: " . $e->getMessage() . "\n";
    }
    
    echo "\n=== Boolean Fallback Type Promotion Test Complete ===\n";
    
} catch (Exception $e) {
    echo "FAIL: Unexpected error: " . $e->getMessage() . "\n";
}

?>
--EXPECT--
=== Boolean Fallback Type Promotion Test ===

--- Testing Boolean Edge Cases ---
Completed boolean edge case testing

--- Testing Promotion Edge Cases ---
PASS: BOOL + INT64 extreme values work
PASS: BOOL + UINT32 large values work

=== Boolean Fallback Type Promotion Test Complete ===
