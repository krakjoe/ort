--TEST--
Test math_core advanced broadcasting - complex shape scenarios
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for advanced broadcasting scenarios

echo "=== Testing Advanced Broadcasting Scenarios ===\n";

// Test 1: Scalar broadcasting (should work)
try {
    $tensor = new ORT\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ORT\Tensor::FLOAT);
    $scalar = new ORT\Tensor\Transient([], [10], ORT\Tensor::FLOAT);
    
    $result = ORT\Math\add($tensor, $scalar);
    echo "PASS: Scalar broadcasting works\n";
} catch (Error $e) {
    echo "FAIL: Scalar broadcasting failed: " . $e->getMessage() . "\n";
}

// Test 2: Vector to matrix broadcasting (1D to 2D)
try {
    $matrix = new ORT\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ORT\Tensor::FLOAT);
    $vector = new ORT\Tensor\Transient([3], [1, 1, 1], ORT\Tensor::FLOAT);
    
    // This should broadcast the vector to each row
    $result = ORT\Math\add($matrix, $vector);
    echo "PASS: Vector to matrix broadcasting works\n";
} catch (Error $e) {
    echo "PARTIAL: Vector to matrix broadcasting: " . $e->getMessage() . "\n";
}

// Test 3: Incompatible shapes for broadcasting
try {
    $tensor_a = new ORT\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ORT\Tensor::FLOAT);
    $tensor_b = new ORT\Tensor\Transient([2, 4], [[1, 2, 3, 4], [5, 6, 7, 8]], ORT\Tensor::FLOAT);
    
    $result = ORT\Math\add($tensor_a, $tensor_b);
    echo "FAIL: Incompatible shapes should be rejected\n";
} catch (ORT\Status\Math\InvalidShape $e) {
    echo "PASS: Incompatible shapes correctly rejected\n";
} catch (Error $e) {
    echo "PARTIAL: Incompatible shapes rejected with: " . get_class($e) . "\n";
}

// Test 4: Complex 3D tensor scenarios
try {
    $tensor_3d = new ORT\Tensor\Persistent("3d", [2, 2, 2], [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ], ORT\Tensor::FLOAT);
    
    $tensor_2d = new ORT\Tensor\Transient([2, 2], [[1, 1], [1, 1]], ORT\Tensor::FLOAT);
    
    // This should broadcast 2D to 3D
    $result = ORT\Math\multiply($tensor_3d, $tensor_2d);
    echo "PASS: 3D tensor operations work\n";
} catch (Error $e) {
    echo "PARTIAL: 3D tensor operations: " . $e->getMessage() . "\n";
}

// Test 5: Edge case - single element tensors
try {
    $single_a = new ORT\Tensor\Transient([1, 1], [[5]], ORT\Tensor::FLOAT);
    $single_b = new ORT\Tensor\Transient([1, 1], [[3]], ORT\Tensor::FLOAT);
    
    $result = ORT\Math\divide($single_a, $single_b);
    echo "PASS: Single element tensor operations work\n";
} catch (Error $e) {
    echo "FAIL: Single element tensor operations failed: " . $e->getMessage() . "\n";
}

echo "Advanced broadcasting tests completed.\n";
?>
--EXPECT--
=== Testing Advanced Broadcasting Scenarios ===
PASS: Scalar broadcasting works
PASS: Vector to matrix broadcasting works
PASS: Incompatible shapes correctly rejected
PASS: 3D tensor operations work
PASS: Single element tensor operations work
Advanced broadcasting tests completed.
