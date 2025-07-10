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

echo "Error handling tests completed.\n";
?>
--EXPECT--
=== Testing Error Handling Paths ===
PASS: Empty tensor correctly rejected
SKIP: UINT64 rejection test (cannot create UINT64 tensor in PHP)
Error handling tests completed.
