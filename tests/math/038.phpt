--TEST--
Math core uncovered functions - targeted axis and validation testing
--EXTENSIONS--
ort
--FILE--
<?php
// Test specific scenarios that might trigger the remaining uncovered functions

// Test vector operations that might need vector validation
try {
    $non_vector = new ORT\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ORT\Tensor::FLOAT32);
    // Try to use operations that might expect vectors
    $result = ORT\Math\sqrt($non_vector); // This should work but might trigger validation
    $data = $result->getData();
    echo "Vector-like operation on matrix: " . $data[0][0] . "\n";
} catch (Throwable $e) {
    echo "Vector validation error: " . $e->getMessage() . "\n";
}

// Test matrix operations that might need matrix validation  
try {
    $non_matrix = new ORT\Tensor\Transient([3], [1, 2, 3], ORT\Tensor::FLOAT32);
    $matrix = new ORT\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ORT\Tensor::FLOAT32);
    // Try matrix multiplication with incompatible shapes
    $result = ORT\Math\matmul($non_matrix, $matrix);
    echo "Unexpected matmul success\n";
} catch (Throwable $e) {
    echo "Expected matrix validation error: " . get_class($e) . "\n";
}

// Test operations that might use type casting validation
try {
    // UINT64 is unsupported and should be detected during tensor creation
    $float_tensor = new ORT\Tensor\Transient([2], [1.0, 2.0], ORT\Tensor::FLOAT32);
    $int_tensor = new ORT\Tensor\Transient([2], [1, 2], ORT\Tensor::INT32);
    $result = ORT\Math\add($int_tensor, $float_tensor);
    $data = $result->getData();
    echo "Mixed type operation: " . $data[0] . "\n";
} catch (Throwable $e) {
    echo "Mixed type error: " . $e->getMessage() . "\n";
}

// Test broadcasitng with very specific shapes to trigger compatibility checks
try {
    $tensor1 = new ORT\Tensor\Transient([1, 3, 1], [[[1], [2], [3]]], ORT\Tensor::FLOAT32);
    $tensor2 = new ORT\Tensor\Transient([2, 1, 4], [[[1, 2, 3, 4]], [[5, 6, 7, 8]]], ORT\Tensor::FLOAT32);
    $result = ORT\Math\add($tensor1, $tensor2);
    $data = $result->getData();
    echo "Complex broadcasting success: " . $data[0][0][0] . "\n";
} catch (Throwable $e) {
    echo "Complex broadcasting error: " . $e->getMessage() . "\n";
}

// Test edge case with 0-dimensional (scalar) operations
try {
    $scalar = new ORT\Tensor\Transient([], [5.0], ORT\Tensor::FLOAT32);
    $vector = new ORT\Tensor\Transient([3], [1, 2, 3], ORT\Tensor::FLOAT32);
    $result = ORT\Math\add($scalar, $vector);
    $data = $result->getData();
    echo "Scalar + vector: " . $data[0] . "\n";
} catch (Throwable $e) {
    echo "Scalar operation error: " . $e->getMessage() . "\n";
}

echo "Targeted validation test completed\n";
?>
--EXPECT--
Vector-like operation on matrix: 1
Expected matrix validation error: ORT\Status\Math\Error
Mixed type operation: 2
Complex broadcasting success: 2
Scalar + vector: 6
Targeted validation test completed
