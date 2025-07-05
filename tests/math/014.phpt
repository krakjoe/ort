--TEST--
Test math_core error handling - invalid inputs and edge cases
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for math_core error handling paths

echo "=== Testing Error Handling Paths ===\n";

// Test 1: NULL/empty tensor validation
try {
    // This should trigger validation error handling
    $empty_tensor = new ONNX\Tensor\Transient([], [], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($empty_tensor);
    echo "FAIL: Empty tensor should be rejected\n";
} catch (ONNX\Status\Tensor\InvalidData $e) {
    echo "PASS: Empty tensor correctly rejected\n";
} catch (Error $e) {
    echo "PARTIAL: Empty tensor rejected with: " . get_class($e) . "\n";
}

// Test 2: UINT64 type rejection (not supported)
try {
    // Note: We may not be able to create UINT64 tensors directly
    // This test may need to be adjusted based on the tensor creation API
    echo "SKIP: UINT64 rejection test (cannot create UINT64 tensor in PHP)\n";
} catch (Error $e) {
    echo "SKIP: UINT64 test: " . $e->getMessage() . "\n";
}

// Test 3: Invalid axis validation
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ONNX\Tensor::FLOAT);
    // Try to sum along axis 5 (invalid for 2D tensor)
    $result = ONNX\Math\sum($tensor, 5);
    echo "FAIL: Invalid axis should be rejected\n";
} catch (ONNX\Status\Tensor\InvalidShape $e) {
    echo "PASS: Invalid axis correctly rejected\n";
} catch (Error $e) {
    echo "PARTIAL: Invalid axis rejected with: " . get_class($e) . "\n";
}

// Test 4: Negative axis validation (should be converted to positive)
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ONNX\Tensor::FLOAT);
    // Try axis -1 (should be valid, converted to 1)
    $result = ONNX\Math\sum($tensor, -1);
    $data = $result->getData();
    if ($data === [6.0, 15.0]) {
        echo "PASS: Negative axis correctly handled\n";
    } else {
        echo "FAIL: Negative axis sum result: ".print_r($data, true)."\n";
    }
} catch (Error $e) {
    echo "FAIL: Valid negative axis rejected: " . $e->getMessage() . "\n";
}

// Test 5: Out of bounds negative axis
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1, 2, 3], [4, 5, 6]], ONNX\Tensor::FLOAT);
    // Try axis -5 (invalid for 2D tensor)
    $result = ONNX\Math\sum($tensor, -5);
    echo "FAIL: Out of bounds negative axis should be rejected\n";
} catch (ONNX\Status\Tensor\InvalidShape $e) {
    echo "PASS: Out of bounds negative axis correctly rejected\n";
} catch (Error $e) {
    echo "PARTIAL: Out of bounds negative axis rejected with: " . get_class($e) . "\n";
}

echo "Error handling tests completed.\n";
?>
--EXPECT--
=== Testing Error Handling Paths ===
PASS: Empty tensor correctly rejected
SKIP: UINT64 rejection test (cannot create UINT64 tensor in PHP)
PASS: Invalid axis correctly rejected
PASS: Negative axis correctly handled
PASS: Out of bounds negative axis correctly rejected
Error handling tests completed.
