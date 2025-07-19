--TEST--
Math core utilities edge cases and validation functions
--EXTENSIONS--
ort
--FILE--
<?php
// Test operations that might trigger uncovered math_core.c functions

// Test broadcast compatibility checks with incompatible shapes
try {
    $tensor_2x3 = new ORT\Tensor\Transient([2, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ORT\Tensor::FLOAT);
    $tensor_3x2 = new ORT\Tensor\Transient([3, 2], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], ORT\Tensor::FLOAT); 
    
    // This should trigger broadcast compatibility checking
    $result = ORT\Math\add($tensor_2x3, $tensor_3x2);
    echo "Unexpected success with incompatible shapes\n";
} catch (Throwable $e) {
    echo "Expected broadcast error: " . get_class($e) . "\n";
}

// Test type casting with various type combinations that might use ort_math_type_can_cast
try {
    $int_tensor = new ORT\Tensor\Transient([2], [10, 20], ORT\Tensor::INT32);
    $float_tensor = new ORT\Tensor\Transient([2], [1.5, 2.5], ORT\Tensor::FLOAT);
    
    // This should trigger type promotion which might use type_can_cast
    $result = ORT\Math\add($int_tensor, $float_tensor);
    $data = $result->getData();
    echo "Type promotion result: " . $data[0] . ", " . $data[1] . "\n";
} catch (Throwable $e) {
    echo "Type promotion error: " . $e->getMessage() . "\n";
}

// Test operations with empty/invalid tensors that might trigger validation
try {
    $empty_tensor = new ORT\Tensor\Transient([0], [], ORT\Tensor::FLOAT);
    $valid_tensor = new ORT\Tensor\Transient([2], [1.0, 2.0], ORT\Tensor::FLOAT);
    
    // This might trigger ort_math_validate_input with empty tensor
    $result = ORT\Math\add($empty_tensor, $valid_tensor);
    echo "Unexpected success with empty tensor\n";
} catch (Throwable $e) {
    echo "Expected empty tensor error: " . get_class($e) . "\n";
}

// Test edge case with 1D tensors for vector operations
try {
    $vector = new ORT\Tensor\Transient([3], [1.0, 4.0, 9.0], ORT\Tensor::FLOAT);
    
    // Use sqrt which is a unary operation
    $result = ORT\Math\sqrt($vector);
    $data = $result->getData();
    echo "Vector sqrt: " . $data[0] . ", " . $data[1] . ", " . $data[2] . "\n";
} catch (Throwable $e) {
    echo "Vector operation error: " . $e->getMessage() . "\n";
}

// Test operations that might trigger string utility functions for error messages
try {
    $matrix_2x2 = new ORT\Tensor\Transient([2, 2], [[1.0, 2.0], [3.0, 4.0]], ORT\Tensor::FLOAT);
    $matrix_3x2 = new ORT\Tensor\Transient([3, 2], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], ORT\Tensor::FLOAT);
    
    // Matrix multiplication with incompatible dimensions
    $result = ORT\Math\matmul($matrix_2x2, $matrix_3x2);
    echo "Unexpected matmul success\n";
} catch (Throwable $e) {
    echo "Expected matmul dimension error: " . get_class($e) . "\n";
}

// Test with different data types to exercise more type promotion paths
try {
    $bool_tensor = new ORT\Tensor\Transient([2], [true, false], ORT\Tensor::BOOL);
    $double_tensor = new ORT\Tensor\Transient([2], [3.14, 2.71], ORT\Tensor::DOUBLE);
    
    // Boolean + Double type promotion
    $result = ORT\Math\add($bool_tensor, $double_tensor);
    $data = $result->getData();
    echo "Bool+Double result: " . $data[0] . ", " . $data[1] . "\n";
} catch (Throwable $e) {
    echo "Bool+Double error: " . $e->getMessage() . "\n";
}

echo "Math core edge cases test completed\n";
?>
--EXPECT--
Expected broadcast error: ORT\Status\Math\InvalidShape
Type promotion result: 11.5, 22.5
Expected empty tensor error: ORT\Status\Tensor\InvalidShape
Vector sqrt: 1, 2, 3
Expected matmul dimension error: ORT\Status\Math\Error
Bool+Double result: 4.14, 2.71
Math core edge cases test completed
