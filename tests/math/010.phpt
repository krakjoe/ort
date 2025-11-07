--TEST--
Test math_core validation functions - axis, matrix, vector validation
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for math validation functions

echo "=== Testing Validation Functions ===\n";

// Test 1: Matrix validation (matmul requires 2D)
try {
    $matrix_a = new ORT\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ORT\Tensor::FLOAT32);
    $matrix_b = new ORT\Tensor\Transient([3, 2], [[1, 2], [3, 4], [5, 6]], ORT\Tensor::FLOAT32);
    $result = ORT\Math\matmul($matrix_a, $matrix_b);
    echo "PASS: Valid 2D matrix multiplication works\n";
} catch (Error $e) {
    echo "FAIL: Valid matrix operation failed: " . $e->getMessage() . "\n";
}

// Test 2: Invalid matrix (3D tensor to matmul)
try {
    $tensor_3d = new ORT\Tensor\Transient([2, 2, 2], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], ORT\Tensor::FLOAT32);
    $matrix = new ORT\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ORT\Tensor::FLOAT32);
    $result = ORT\Math\matmul($tensor_3d, $matrix);
    echo "FAIL: 3D tensor should be rejected for matmul\n";
} catch (ORT\Status\Math\Error $e) {
    echo "PASS: 3D tensor correctly rejected for matmul\n";
} catch (Error $e) {
    echo "PARTIAL: Got exception for 3D tensor: " . get_class($e) . "\n";
}

// Test 3: Invalid matrix (1D vector to matmul)
try {
    $vector = new ORT\Tensor\Transient([4], [1, 2, 3, 4], ORT\Tensor::FLOAT32);
    $matrix = new ORT\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ORT\Tensor::FLOAT32);
    $result = ORT\Math\matmul($vector, $matrix);
    echo "FAIL: 1D vector should be rejected for matmul\n";
} catch (ORT\Status\Math\Error $e) {
    echo "PASS: 1D vector correctly rejected for matmul\n";
} catch (Error $e) {
    echo "PARTIAL: Got exception for 1D vector: " . get_class($e) . "\n";
}

// Test 4: Matrix dimension mismatch
try {
    $matrix_a = new ORT\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ORT\Tensor::FLOAT32);
    $matrix_b = new ORT\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ORT\Tensor::FLOAT32); // Wrong inner dimension
    $result = ORT\Math\matmul($matrix_a, $matrix_b);
    echo "FAIL: Incompatible matrix dimensions should be rejected\n";
} catch (ORT\Status\Math\Error $e) {
    echo "PASS: Matrix dimension mismatch correctly rejected\n";
} catch (Error $e) {
    echo "PARTIAL: Got exception for dimension mismatch: " . get_class($e) . "\n";
}

// Test 5: Empty tensor validation
try {
    $empty_tensor = new ORT\Tensor\Transient([0], [], ORT\Tensor::FLOAT32);
    $result = ORT\Math\sqrt($empty_tensor);
    echo "FAIL: Empty tensor should be rejected\n";
} catch (ORT\Status\Tensor\InvalidShape $e) {
    echo "PASS: Empty tensor correctly rejected\n";
} catch (Error $e) {
    echo "PARTIAL: Got exception for empty tensor: " . get_class($e) . "\n";
}

// Test 6: Null data validation (this tests the internal validation)
// Note: This is harder to test from PHP since we can't easily create invalid tensors
echo "INFO: Internal null data validation tested through C unit tests\n";

echo "Validation tests completed.\n";
?>
--EXPECT--
=== Testing Validation Functions ===
PASS: Valid 2D matrix multiplication works
PASS: 3D tensor correctly rejected for matmul
PASS: 1D vector correctly rejected for matmul
PASS: Matrix dimension mismatch correctly rejected
PASS: Empty tensor correctly rejected
INFO: Internal null data validation tested through C unit tests
Validation tests completed.
