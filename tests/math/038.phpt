--TEST--
Math core uncovered functions - targeted axis and validation testing
--EXTENSIONS--
ort
--FILE--
<?php
// Test specific scenarios that might trigger the remaining uncovered functions

// Test axis validation - try to create conditions that need axis checking
try {
    $tensor = new ONNX\Tensor\Transient([2, 3, 4], 
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 
         [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]], 
        ONNX\Tensor::FLOAT);
    
    // Try sum with out-of-bounds axis to trigger validation
    $result = ONNX\Math\sum($tensor, 5); // axis 5 is out of bounds for 3D tensor
    echo "Unexpected success with invalid axis\n";
} catch (Throwable $e) {
    echo "Expected axis error: " . get_class($e) . "\n";
}

// Test negative axis to trigger axis validation
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, -5); // axis -5 is out of bounds
    echo "Unexpected success with negative axis\n";
} catch (Throwable $e) {
    echo "Expected negative axis error: " . get_class($e) . "\n";
}

// Test vector operations that might need vector validation
try {
    $non_vector = new ONNX\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ONNX\Tensor::FLOAT);
    // Try to use operations that might expect vectors
    $result = ONNX\Math\sqrt($non_vector); // This should work but might trigger validation
    $data = $result->getData();
    echo "Vector-like operation on matrix: " . $data[0][0] . "\n";
} catch (Throwable $e) {
    echo "Vector validation error: " . $e->getMessage() . "\n";
}

// Test matrix operations that might need matrix validation  
try {
    $non_matrix = new ONNX\Tensor\Transient([3], [1, 2, 3], ONNX\Tensor::FLOAT);
    $matrix = new ONNX\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ONNX\Tensor::FLOAT);
    // Try matrix multiplication with incompatible shapes
    $result = ONNX\Math\matmul($non_matrix, $matrix);
    echo "Unexpected matmul success\n";
} catch (Throwable $e) {
    echo "Expected matrix validation error: " . get_class($e) . "\n";
}

// Test operations that might use type casting validation
try {
    // UINT64 is unsupported and should be detected during tensor creation
    $float_tensor = new ONNX\Tensor\Transient([2], [1.0, 2.0], ONNX\Tensor::FLOAT);
    $int_tensor = new ONNX\Tensor\Transient([2], [1, 2], ONNX\Tensor::INT32);
    $result = ONNX\Math\add($int_tensor, $float_tensor);
    $data = $result->getData();
    echo "Mixed type operation: " . $data[0] . "\n";
} catch (Throwable $e) {
    echo "Mixed type error: " . $e->getMessage() . "\n";
}

// Test broadcasitng with very specific shapes to trigger compatibility checks
try {
    $tensor1 = new ONNX\Tensor\Transient([1, 3, 1], [[[1], [2], [3]]], ONNX\Tensor::FLOAT);
    $tensor2 = new ONNX\Tensor\Transient([2, 1, 4], [[[1, 2, 3, 4]], [[5, 6, 7, 8]]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\add($tensor1, $tensor2);
    $data = $result->getData();
    echo "Complex broadcasting success: " . $data[0][0][0] . "\n";
} catch (Throwable $e) {
    echo "Complex broadcasting error: " . $e->getMessage() . "\n";
}

// Test edge case with 0-dimensional (scalar) operations
try {
    $scalar = new ONNX\Tensor\Transient([], [5.0], ONNX\Tensor::FLOAT);
    $vector = new ONNX\Tensor\Transient([3], [1, 2, 3], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\add($scalar, $vector);
    $data = $result->getData();
    echo "Scalar + vector: " . $data[0] . "\n";
} catch (Throwable $e) {
    echo "Scalar operation error: " . $e->getMessage() . "\n";
}

echo "Targeted validation test completed\n";
?>
--EXPECT--
Expected axis error: ONNX\Status\Tensor\InvalidShape
Expected negative axis error: ONNX\Status\Tensor\InvalidShape
Vector-like operation on matrix: 1
Expected matrix validation error: ONNX\Status\Math\Error
Mixed type operation: 2
Complex broadcasting success: 2
Scalar + vector: 6
Targeted validation test completed
