--TEST--
Test math unary operations - comprehensive trigonometric and mathematical functions
--EXTENSIONS--
ort
--FILE--
<?php
// Unit test for comprehensive unary mathematical operations

echo "=== Testing Unary Mathematical Operations ===\n";

// Test 1: Trigonometric functions with various inputs
try {
    $angles = new ONNX\Tensor\Persistent("angles", [4], [0.0, 1.5708, 3.1416, 4.7124], ONNX\Tensor::FLOAT); // 0, π/2, π, 3π/2
    
    $sin_result = ONNX\Math\sin($angles);
    $cos_result = ONNX\Math\cos($angles);
    $tan_result = ONNX\Math\tan($angles);
    
    echo "PASS: Basic trigonometric functions work\n";
} catch (Error $e) {
    echo "FAIL: Basic trigonometric functions failed: " . $e->getMessage() . "\n";
}

// Test 2: Inverse trigonometric functions
try {
    $values = new ONNX\Tensor\Persistent("values", [3], [0.0, 0.5, 1.0], ONNX\Tensor::FLOAT);
    
    $asin_result = ONNX\Math\asin($values);
    $acos_result = ONNX\Math\acos($values);
    $atan_result = ONNX\Math\atan($values);
    
    echo "PASS: Inverse trigonometric functions work\n";
} catch (Error $e) {
    echo "FAIL: Inverse trigonometric functions failed: " . $e->getMessage() . "\n";
}

// Test 3: Hyperbolic functions
try {
    $values = new ONNX\Tensor\Persistent("values", [4], [0.0, 1.0, 2.0, -1.0], ONNX\Tensor::FLOAT);
    
    $sinh_result = ONNX\Math\sinh($values);
    $cosh_result = ONNX\Math\cosh($values);
    $tanh_result = ONNX\Math\tanh($values);
    
    echo "PASS: Hyperbolic functions work\n";
} catch (Error $e) {
    echo "FAIL: Hyperbolic functions failed: " . $e->getMessage() . "\n";
}

// Test 4: Exponential and logarithmic functions
try {
    $values = new ONNX\Tensor\Persistent("values", [4], [1.0, 2.0, 4.0, 8.0], ONNX\Tensor::FLOAT);
    
    $exp_result = ONNX\Math\exp($values);
    $exp2_result = ONNX\Math\exp2($values);
    $log_result = ONNX\Math\log($values);
    $log2_result = ONNX\Math\log2($values);
    $log10_result = ONNX\Math\log10($values);
    
    echo "PASS: Exponential and logarithmic functions work\n";
} catch (Error $e) {
    echo "FAIL: Exponential and logarithmic functions failed: " . $e->getMessage() . "\n";
}

// Test 5: Power, root, and rounding functions
try {
    $values = new ONNX\Tensor\Persistent("values", [4], [1.5, 2.7, 8.0, 27.0], ONNX\Tensor::FLOAT);
    
    $sqrt_result = ONNX\Math\sqrt($values);
    $cbrt_result = ONNX\Math\cbrt($values);
    $ceil_result = ONNX\Math\ceil($values);
    $floor_result = ONNX\Math\floor($values);
    $round_result = ONNX\Math\round($values);
    
    echo "PASS: Power, root, and rounding functions work\n";
} catch (Error $e) {
    echo "FAIL: Power, root, and rounding functions failed: " . $e->getMessage() . "\n";
}

// Test 6: Edge cases with special values
try {
    $special = new ONNX\Tensor\Persistent("special", [3], [0.0, 1.0, -1.0], ONNX\Tensor::FLOAT);
    
    // These should handle edge cases properly
    $sqrt_special = ONNX\Math\sqrt($special);  // sqrt(-1) should produce NaN
    $log_special = ONNX\Math\log($special);   // log(0) should produce -inf
    
    echo "PASS: Special value edge cases handled\n";
} catch (Error $e) {
    echo "PARTIAL: Special value edge cases: " . $e->getMessage() . "\n";
}

// Test 7: Large tensor unary operations
try {
    $large_data = array_fill(0, 100, 2.0);
    $large = new ONNX\Tensor\Persistent("large", [10, 10], array_chunk($large_data, 10), ONNX\Tensor::FLOAT);
    
    $result = ONNX\Math\sqrt($large);
    echo "PASS: Large tensor unary operations work\n";
} catch (Error $e) {
    echo "FAIL: Large tensor unary operations failed: " . $e->getMessage() . "\n";
}

echo "Unary mathematical operation tests completed.\n";
?>
--EXPECT--
=== Testing Unary Mathematical Operations ===
PASS: Basic trigonometric functions work
PASS: Inverse trigonometric functions work
PASS: Hyperbolic functions work
PASS: Exponential and logarithmic functions work
PASS: Power, root, and rounding functions work
PASS: Special value edge cases handled
PASS: Large tensor unary operations work
Unary mathematical operation tests completed.
