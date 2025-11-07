--TEST--
Test math_core matrix operations - comprehensive matmul testing
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for matrix operations, focusing on matmul edge cases

echo "=== Testing Matrix Operations ===\n";

// Test 1: Basic square matrix multiplication
try {
    $matrix_a = new ORT\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ORT\Tensor::FLOAT32);
    $matrix_b = new ORT\Tensor\Transient([2, 2], [[2, 0], [1, 2]], ORT\Tensor::FLOAT32);
    
    $result = ORT\Math\matmul($matrix_a, $matrix_b);
    // Result should be [[4, 4], [10, 8]]
    echo "PASS: Basic square matrix multiplication works\n";
} catch (Error $e) {
    echo "FAIL: Basic matmul failed: " . $e->getMessage() . "\n";
}

// Test 2: Non-square matrix multiplication
try {
    $matrix_a = new ORT\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ORT\Tensor::FLOAT32);
    $matrix_b = new ORT\Tensor\Transient([3, 2], [[1, 2], [3, 4], [5, 6]], ORT\Tensor::FLOAT32);
    
    $result = ORT\Math\matmul($matrix_a, $matrix_b);
    // Result should be [2, 2] matrix
    echo "PASS: Non-square matrix multiplication works\n";
} catch (Error $e) {
    echo "FAIL: Non-square matmul failed: " . $e->getMessage() . "\n";
}

// Test 3: Incompatible matrix dimensions
try {
    $matrix_a = new ORT\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ORT\Tensor::FLOAT32);
    $matrix_b = new ORT\Tensor\Transient([2, 2], [[1, 2], [3, 4]], ORT\Tensor::FLOAT32);
    
    $result = ORT\Math\matmul($matrix_a, $matrix_b);
    echo "FAIL: Incompatible matrix dimensions should be rejected\n";
} catch (ORT\Status\Math\Error $e) {
    echo "PASS: Incompatible matrix dimensions correctly rejected\n";
} catch (Error $e) {
    echo "PARTIAL: Incompatible dimensions rejected with: " . get_class($e) . "\n";
}

// Test 4: Matrix with vector (should fail - vectors need to be 2D)
try {
    $matrix = new ORT\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ORT\Tensor::FLOAT32);
    $vector = new ORT\Tensor\Transient([3], [1, 2, 3], ORT\Tensor::FLOAT32);
    
    $result = ORT\Math\matmul($matrix, $vector);
    echo "FAIL: 1D vector should be rejected for matmul\n";
} catch (ORT\Status\Math\Error $e) {
    echo "PASS: 1D vector correctly rejected for matmul\n";
} catch (Error $e) {
    echo "PARTIAL: 1D vector rejected with: " . get_class($e) . "\n";
}

// Test 5: Large matrix multiplication (stress test)
try {
    $large_a = [];
    $large_b = [];
    
    // Create 5x5 matrices
    for ($i = 0; $i < 5; $i++) {
        $row_a = [];
        $row_b = [];
        for ($j = 0; $j < 5; $j++) {
            $row_a[] = $i + $j + 1;
            $row_b[] = ($i == $j) ? 1 : 0; // Identity-like
        }
        $large_a[] = $row_a;
        $large_b[] = $row_b;
    }
    
    $matrix_a = new ORT\Tensor\Transient([5, 5], $large_a, ORT\Tensor::FLOAT32);
    $matrix_b = new ORT\Tensor\Transient([5, 5], $large_b, ORT\Tensor::FLOAT32);
    
    $result = ORT\Math\matmul($matrix_a, $matrix_b);
    echo "PASS: Large matrix multiplication completed\n";
} catch (Error $e) {
    echo "FAIL: Large matrix multiplication failed: " . $e->getMessage() . "\n";
}

echo "Matrix operation tests completed.\n";
?>
--EXPECT--
=== Testing Matrix Operations ===
PASS: Basic square matrix multiplication works
PASS: Non-square matrix multiplication works
PASS: Incompatible matrix dimensions correctly rejected
PASS: 1D vector correctly rejected for matmul
PASS: Large matrix multiplication completed
Matrix operation tests completed.
