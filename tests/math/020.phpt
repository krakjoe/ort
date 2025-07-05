--TEST--
Test math_core reduction operations - sum with various axes and options
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for reduction operations, focusing on sum with different axes

echo "=== Testing Reduction Operations ===\n";

// Test 1: Sum all elements (no axis)
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor);
    // Should sum all elements: 1+2+3+4+5+6 = 21
    echo "PASS: Sum all elements works\n";
} catch (Error $e) {
    echo "FAIL: Sum all elements failed: " . $e->getMessage() . "\n";
}

// Test 2: Sum along axis 0 (rows)
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, 0);
    // Should produce [5, 7, 9] (sum each column)
    echo "PASS: Sum along axis 0 works\n";
} catch (Error $e) {
    echo "FAIL: Sum along axis 0 failed: " . $e->getMessage() . "\n";
}

// Test 3: Sum along axis 1 (columns)
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, 1);
    // Should produce [6, 15] (sum each row)
    echo "PASS: Sum along axis 1 works\n";
} catch (Error $e) {
    echo "FAIL: Sum along axis 1 failed: " . $e->getMessage() . "\n";
}

// Test 4: Sum with keepdims=true
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, 1, true);
    // Should produce [2, 1] shape instead of [2]
    echo "PASS: Sum with keepdims works\n";
} catch (Error $e) {
    echo "FAIL: Sum with keepdims failed: " . $e->getMessage() . "\n";
}

// Test 5: Sum along negative axis
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, -1);  // Should be equivalent to axis 1
    echo "PASS: Sum along negative axis works\n";
} catch (Error $e) {
    echo "FAIL: Sum along negative axis failed: " . $e->getMessage() . "\n";
}

// Test 6: 3D tensor reduction
try {
    $tensor_3d = new ONNX\Tensor\Persistent("3d", [2, 2, 2], [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ], ONNX\Tensor::FLOAT);
    
    $result = ONNX\Math\sum($tensor_3d, 0);  // Sum along first axis
    echo "PASS: 3D tensor reduction works\n";
} catch (Error $e) {
    echo "FAIL: 3D tensor reduction failed: " . $e->getMessage() . "\n";
}

// Test 7: Single element tensor
try {
    $tensor = new ONNX\Tensor\Transient([1, 1], [[42]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor);
    echo "PASS: Single element tensor reduction works\n";
} catch (Error $e) {
    echo "FAIL: Single element tensor reduction failed: " . $e->getMessage() . "\n";
}

echo "Reduction operation tests completed.\n";
?>
--EXPECT--
=== Testing Reduction Operations ===
PASS: Sum all elements works
PASS: Sum along axis 0 works
PASS: Sum along axis 1 works
PASS: Sum with keepdims works
PASS: Sum along negative axis works
PASS: 3D tensor reduction works
PASS: Single element tensor reduction works
Reduction operation tests completed.
