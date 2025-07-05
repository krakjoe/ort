--TEST--
Test math_core result management - create and free result structures
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for math_core result management functions

echo "=== Testing Result Management ===\n";

// Test 1: Basic result creation and memory handling
try {
    $tensor_a = new ONNX\Tensor\Persistent("a", [2, 2], [[1, 2], [3, 4]], ONNX\Tensor::FLOAT);
    $tensor_b = new ONNX\Tensor\Persistent("b", [2, 2], [[1, 1], [1, 1]], ONNX\Tensor::FLOAT);
    
    // This should create and properly manage result structures
    $result = ONNX\Math\add($tensor_a, $tensor_b);
    
    if ($result instanceof ONNX\Tensor) {
        echo "PASS: Result properly created as tensor\n";
    } else {
        echo "FAIL: Result is not a tensor: " . gettype($result) . "\n";
    }
} catch (Error $e) {
    echo "FAIL: Result creation failed: " . $e->getMessage() . "\n";
}

// Test 2: Multiple operations to test memory management
try {
    $tensor = new ONNX\Tensor\Persistent("test", [3, 3], [
        [1, 2, 3], 
        [4, 5, 6], 
        [7, 8, 9]
    ], ONNX\Tensor::FLOAT);
    
    // Chain operations to test result management
    $result1 = ONNX\Math\sqrt($tensor);
    $result2 = ONNX\Math\multiply($result1, $result1);  // Should be close to original
    $result3 = ONNX\Math\add($result2, $tensor);
    
    echo "PASS: Multiple result operations completed\n";
} catch (Error $e) {
    echo "FAIL: Multiple result operations failed: " . $e->getMessage() . "\n";
}

// Test 3: Test scalar operations result management
try {
    $tensor = new ONNX\Tensor\Persistent("scalar_test", [2, 3], [
        [1, 2, 3], 
        [4, 5, 6]
    ], ONNX\Tensor::FLOAT);
    
    // Scalar operations
    $result1 = ONNX\Math\add($tensor, 10);
    $result2 = ONNX\Math\multiply($result1, 0.5);
    $result3 = ONNX\Math\subtract($result2, 5);
    
    echo "PASS: Scalar operation results managed properly\n";
} catch (Error $e) {
    echo "FAIL: Scalar operation result management failed: " . $e->getMessage() . "\n";
}

echo "Result management tests completed.\n";
?>
--EXPECT--
=== Testing Result Management ===
PASS: Result properly created as tensor
PASS: Multiple result operations completed
PASS: Scalar operation results managed properly
Result management tests completed.
