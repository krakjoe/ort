--TEST--
Test math_core broadcasting functionality - ort_math_broadcast_calculate
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for broadcasting compatibility checking

echo "=== Testing Broadcasting Compatibility ===\n";

// Test 1: Compatible shapes (same dimensions)
try {
    $tensor_a1 = new ORT\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ORT\Tensor::FLOAT);
    $tensor_b1 = new ORT\Tensor\Transient([2, 3], [[1, 1, 1], [2, 2, 2]], ORT\Tensor::FLOAT);
    $result = ORT\Math\add($tensor_a1, $tensor_b1);
    echo "PASS: Compatible shapes [2,3] + [2,3] work\n";
} catch (Error $e) {
    echo "FAIL: Compatible shapes should work: " . $e->getMessage() . "\n";
}

// Test 2: Incompatible shapes (should fail)
try {
    $tensor_a2 = new ORT\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ORT\Tensor::FLOAT);
    $tensor_b2 = new ORT\Tensor\Transient([3, 2], [[1, 2], [3, 4], [5, 6]], ORT\Tensor::FLOAT);
    $result = ORT\Math\add($tensor_a2, $tensor_b2);
    echo "FAIL: Incompatible shapes [2,3] + [3,2] should fail\n";
} catch (ORT\Status\Math\InvalidShape $e) {
    echo "PASS: Incompatible shapes correctly rejected\n";
} catch (Error $e) {
    echo "PASS: Incompatible shapes correctly rejected\n";
}

// Test 3: Different number of dimensions (should fail currently)
try {
    $tensor_a3 = new ORT\Tensor\Transient([6], [1, 2, 3, 4, 5, 6], ORT\Tensor::FLOAT);
    $tensor_b3 = new ORT\Tensor\Transient([2, 3], [[1, 1, 1], [2, 2, 2]], ORT\Tensor::FLOAT);
    $result = ORT\Math\add($tensor_a3, $tensor_b3);
    echo "FAIL: Different dimensions [6] + [2,3] should fail\n";
} catch (ORT\Status\Math\InvalidShape $e) {
    echo "PASS: Different dimensions correctly rejected\n";
} catch (Error $e) {
    echo "PASS: Different dimensions correctly rejected\n";
}

echo "Broadcasting tests completed.\n";
?>
--EXPECT--
=== Testing Broadcasting Compatibility ===
PASS: Compatible shapes [2,3] + [2,3] work
PASS: Incompatible shapes correctly rejected
PASS: Different dimensions correctly rejected
Broadcasting tests completed.
