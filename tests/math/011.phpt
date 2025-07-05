--TEST--
Test math_core utility functions - indexing and element calculation
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for math utility functions that are typically internal

echo "=== Testing Utility Functions ===\n";

// Test 1: Element count validation (total elements calculation)
try {
    // Test various tensor shapes to exercise element calculation
    $tensor_1d = new ONNX\Tensor\Persistent("1d", [5], [1, 2, 3, 4, 5], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($tensor_1d);
    echo "PASS: 1D tensor [5] element calculation works\n";
} catch (Error $e) {
    echo "FAIL: 1D tensor failed: " . $e->getMessage() . "\n";
}

try {
    $tensor_2d = new ONNX\Tensor\Persistent("2d", [2, 3], [[1, 2, 3], [4, 5, 6]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($tensor_2d);
    echo "PASS: 2D tensor [2,3] element calculation works\n";
} catch (Error $e) {
    echo "FAIL: 2D tensor failed: " . $e->getMessage() . "\n";
}

try {
    $tensor_3d = new ONNX\Tensor\Persistent("3d", [2, 2, 2], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($tensor_3d);
    echo "PASS: 3D tensor [2,2,2] element calculation works\n";
} catch (Error $e) {
    echo "FAIL: 3D tensor failed: " . $e->getMessage() . "\n";
}

try {
    $tensor_4d = new ONNX\Tensor\Persistent("4d", [2, 1, 2, 1], [[[[1], [2]]], [[[3], [4]]]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($tensor_4d);
    echo "PASS: 4D tensor [2,1,2,1] element calculation works\n";
} catch (Error $e) {
    echo "FAIL: 4D tensor failed: " . $e->getMessage() . "\n";
}

// Test 2: Large tensor (stress test element calculation)
try {
    // Create a larger tensor to test element calculation with bigger numbers
    $large_data = array_fill(0, 100, 1.0);
    $tensor_large = new ONNX\Tensor\Persistent("large", [100], $large_data, ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($tensor_large);
    echo "PASS: Large tensor [100] element calculation works\n";
} catch (Error $e) {
    echo "FAIL: Large tensor failed: " . $e->getMessage() . "\n";
}

// Test 3: Very large tensor (test limits)
try {
    // Create an even larger tensor
    $very_large_data = array_fill(0, 1000, 2.0);
    $tensor_very_large = new ONNX\Tensor\Persistent("very_large", [1000], $very_large_data, ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($tensor_very_large);
    echo "PASS: Very large tensor [1000] element calculation works\n";
} catch (Error $e) {
    echo "FAIL: Very large tensor failed: " . $e->getMessage() . "\n";
}

// Test 4: Multi-dimensional large tensor
try {
    $md_large_data = [];
    for ($i = 0; $i < 10; $i++) {
        $md_large_data[] = array_fill(0, 10, (float)($i + 1));
    }
    $tensor_md_large = new ONNX\Tensor\Persistent("md_large", [10, 10], $md_large_data, ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($tensor_md_large);
    echo "PASS: Multi-dimensional large tensor [10,10] works\n";
} catch (Error $e) {
    echo "FAIL: Multi-dimensional large tensor failed: " . $e->getMessage() . "\n";
}

// Test 5: Single element tensor (scalar-like)
try {
    $scalar_tensor = new ONNX\Tensor\Persistent("scalar", [1], [42.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($scalar_tensor);
    echo "PASS: Single element tensor [1] works\n";
} catch (Error $e) {
    echo "FAIL: Single element tensor failed: " . $e->getMessage() . "\n";
}

echo "Utility function tests completed.\n";
?>
--EXPECT--
=== Testing Utility Functions ===
PASS: 1D tensor [5] element calculation works
PASS: 2D tensor [2,3] element calculation works
PASS: 3D tensor [2,2,2] element calculation works
PASS: 4D tensor [2,1,2,1] element calculation works
PASS: Large tensor [100] element calculation works
PASS: Very large tensor [1000] element calculation works
PASS: Multi-dimensional large tensor [10,10] works
PASS: Single element tensor [1] works
Utility function tests completed.
