--TEST--
Test math_core string utilities - shape and type to string conversion
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for string utility functions used in error messages

echo "=== Testing String Utility Functions ===\n";

// These functions are typically called internally for error messages
// We test them indirectly by triggering errors that should use them

// Test 1: Shape mismatch error (should call ort_math_string_shape_to_string)
try {
    $tensor_a = new ORT\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ORT\Tensor::FLOAT32);
    $tensor_b = new ORT\Tensor\Transient([3, 2], [[1, 2], [3, 4], [5, 6]], ORT\Tensor::FLOAT32);
    $result = ORT\Math\add($tensor_a, $tensor_b);
    echo "FAIL: Shape mismatch should trigger error\n";
} catch (ORT\Status\Math\InvalidShape $e) {
    echo "PASS: Shape mismatch error triggered (tests shape string utility)\n";
    // Check if error message contains useful shape information
    $message = $e->getMessage();
    if (strpos($message, 'shape') !== false || strpos($message, 'dimension') !== false) {
        echo "PASS: Error message contains shape information\n";
    } else {
        echo "INFO: Error message: " . $message . "\n";
    }
} catch (Error $e) {
    echo "PARTIAL: Got error but wrong type: " . get_class($e) . " - " . $e->getMessage() . "\n";
}

// Test 2: Type incompatibility error (should call ort_math_string_type_to_string)
// Note: Current implementation might not have truly incompatible types, 
// but we can test by creating scenarios that exercise type checking
try {
    // Try operations that might trigger type validation
    $tensor_float = new ORT\Tensor\Transient([2], [1.0, 2.0], ORT\Tensor::FLOAT32);
    $tensor_bool = new ORT\Tensor\Transient([2], [true, false], ORT\Tensor::BOOL);
    $result = ORT\Math\add($tensor_float, $tensor_bool);
    echo "INFO: FLOAT + BOOL operation result: success\n";
} catch (ORT\Status\Math\InvalidType $e) {
    echo "PASS: Type incompatibility error triggered (tests type string utility)\n";
    $message = $e->getMessage();
    if (strpos($message, 'type') !== false) {
        echo "PASS: Error message contains type information\n";
    } else {
        echo "INFO: Error message: " . $message . "\n";
    }
} catch (Error $e) {
    echo "INFO: Got different error: " . get_class($e) . " - " . $e->getMessage() . "\n";
}

// Test 3: Matrix operation with wrong dimensions (should trigger detailed error)
try {
    $matrix_wrong = new ORT\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ORT\Tensor::FLOAT32);
    $matrix_incompatible = new ORT\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ORT\Tensor::FLOAT32);
    $result = ORT\Math\matmul($matrix_wrong, $matrix_incompatible);
    echo "FAIL: Matrix dimension mismatch should trigger error\n";
} catch (ORT\Status\Math\Error $e) {
    echo "PASS: Matrix operation error triggered (tests error formatting)\n";
    $message = $e->getMessage();
    if (strpos($message, 'matrix') !== false || strpos($message, 'dimension') !== false) {
        echo "PASS: Error message contains matrix/dimension information\n";
    } else {
        echo "INFO: Error message: " . $message . "\n";
    }
} catch (Error $e) {
    echo "PARTIAL: Got error: " . get_class($e) . " - " . $e->getMessage() . "\n";
}

// Test 4: Try to trigger validation error for 3D tensor in matrix operation
try {
    $tensor_3d = new ORT\Tensor\Transient([2, 2, 2], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], ORT\Tensor::FLOAT32);
    $matrix_2d = new ORT\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ORT\Tensor::FLOAT32);
    $result = ORT\Math\matmul($tensor_3d, $matrix_2d);
    echo "FAIL: 3D tensor should be rejected for matrix operation\n";
} catch (Error $e) {
    echo "PASS: 3D tensor rejected (tests dimension validation)\n";
    $message = $e->getMessage();
    if (strpos($message, '3') !== false || strpos($message, '2D') !== false || strpos($message, 'matrix') !== false || strpos($message, 'batch') !== false || strpos($message, 'dimension') !== false) {
        echo "PASS: Error message contains dimension details\n";
    } else {
        echo "INFO: Error message: " . $message . "\n";
    }
}

// Test 5: Test various data types in error scenarios
$types = [
    ['name' => 'INT8', 'type' => ORT\Tensor::INT8, 'data' => [1, 2]],
    ['name' => 'INT16', 'type' => ORT\Tensor::INT16, 'data' => [1, 2]],
    ['name' => 'INT32', 'type' => ORT\Tensor::INT32, 'data' => [1, 2]],
    ['name' => 'UINT8', 'type' => ORT\Tensor::UINT8, 'data' => [1, 2]],
    ['name' => 'FLOAT64', 'type' => ORT\Tensor::FLOAT64, 'data' => [1.0, 2.0]]
];

$successful_types = 0;
foreach ($types as $type_info) {
    try {
        $tensor = new ORT\Tensor\Transient([2], $type_info['data'], $type_info['type']);
        $result = ORT\Math\add($tensor, $tensor); // Use add instead of sqrt
        $successful_types++;
    } catch (Error $e) {
        echo "INFO: Type " . $type_info['name'] . " failed: " . get_class($e) . "\n";
    }
}
echo "PASS: $successful_types data types successfully processed\n";

echo "String utility tests completed.\n";
?>
--EXPECT--
=== Testing String Utility Functions ===
PASS: Shape mismatch error triggered (tests shape string utility)
PASS: Error message contains shape information
INFO: FLOAT + BOOL operation result: success
PASS: Matrix operation error triggered (tests error formatting)
PASS: Error message contains matrix/dimension information
PASS: 3D tensor rejected (tests dimension validation)
PASS: Error message contains dimension details
PASS: 5 data types successfully processed
String utility tests completed.
