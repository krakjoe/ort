--TEST--
Test math_core type promotion - ort_math_type_promote
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for type promotion system

echo "=== Testing Type Promotion ===\n";

// Test 1: Same types (should keep type)
try {
    $tensor_a1 = new ONNX\Tensor\Transient([2], [1.0, 2.0], ONNX\Tensor::FLOAT);
    $tensor_b1 = new ONNX\Tensor\Transient([2], [3.0, 4.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\add($tensor_a1, $tensor_b1);
    echo "PASS: FLOAT + FLOAT operation works\n";
} catch (Error $e) {
    echo "FAIL: Same type operation failed: " . $e->getMessage() . "\n";
}

// Test 2: Float promotion (FLOAT -> DOUBLE)
try {
    $tensor_a2 = new ONNX\Tensor\Transient([2], [1.0, 2.0], ONNX\Tensor::FLOAT);
    $tensor_b2 = new ONNX\Tensor\Transient([2], [3.0, 4.0], ONNX\Tensor::DOUBLE);
    $result = ONNX\Math\add($tensor_a2, $tensor_b2);
    echo "PASS: FLOAT + DOUBLE promotion works\n";
} catch (Error $e) {
    echo "FAIL: Float promotion failed: " . $e->getMessage() . "\n";
}

// Test 3: Integer types
try {
    $tensor_a3 = new ONNX\Tensor\Transient([2], [1, 2], ONNX\Tensor::INT32);
    $tensor_b3 = new ONNX\Tensor\Transient([2], [3, 4], ONNX\Tensor::INT32);
    $result = ONNX\Math\add($tensor_a3, $tensor_b3);
    echo "PASS: INT32 + INT32 operation works\n";
} catch (Error $e) {
    echo "FAIL: Integer operation failed: " . $e->getMessage() . "\n";
}

// Test 4: Mixed integer promotion
try {
    $tensor_a4 = new ONNX\Tensor\Transient([2], [1, 2], ONNX\Tensor::INT16);
    $tensor_b4 = new ONNX\Tensor\Transient([2], [3, 4], ONNX\Tensor::INT32);
    $result = ONNX\Math\add($tensor_a4, $tensor_b4);
    echo "PASS: INT16 + INT32 promotion works\n";
} catch (Error $e) {
    echo "FAIL: Integer promotion failed: " . $e->getMessage() . "\n";
}

// Test 5: Boolean with numeric (should promote to numeric)
try {
    $tensor_a5 = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
    $tensor_b5 = new ONNX\Tensor\Transient([2], [1, 2], ONNX\Tensor::INT32);
    $result = ONNX\Math\add($tensor_a5, $tensor_b5);
    echo "PASS: BOOL + INT32 promotion works\n";
} catch (Error $e) {
    echo "INFO: BOOL + INT32 not supported: " . get_class($e) . "\n";
}

// Test 6: Incompatible types (if any exist)
try {
    // This should test an actually incompatible combination if any exist
    $tensor_a6 = new ONNX\Tensor\Transient([2], [1, 2], ONNX\Tensor::UINT8);
    $tensor_b6 = new ONNX\Tensor\Transient([2], [3, 4], ONNX\Tensor::UINT32);
    $result = ONNX\Math\add($tensor_a6, $tensor_b6);
    echo "PASS: UINT8 + UINT32 promotion works\n";
} catch (Error $e) {
    echo "INFO: UINT8 + UINT32 promotion: " . get_class($e) . "\n";
}

echo "Type promotion tests completed.\n";
?>
--EXPECT--
=== Testing Type Promotion ===
PASS: FLOAT + FLOAT operation works
PASS: FLOAT + DOUBLE promotion works
PASS: INT32 + INT32 operation works
PASS: INT16 + INT32 promotion works
PASS: BOOL + INT32 promotion works
PASS: UINT8 + UINT32 promotion works
Type promotion tests completed.
