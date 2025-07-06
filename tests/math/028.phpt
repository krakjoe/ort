--TEST--
Test unary math functions - comprehensive coverage  
--EXTENSIONS--
ort
--FILE--
<?php
echo "=== Testing Unary Math Functions - Comprehensive Coverage ===\n";

// Test 1: More trigonometric functions coverage
try {
    $tensor = new ONNX\Tensor\Transient([4], [0.0, 0.5, 1.0, -0.5], ONNX\Tensor::FLOAT);
    
    // Test inverse trig functions
    $result = ONNX\Math\asin($tensor);
    echo "PASS: asin function works\n";
    
    $result = ONNX\Math\acos($tensor);
    echo "PASS: acos function works\n";
    
    $result = ONNX\Math\atan($tensor);
    echo "PASS: atan function works\n";
} catch (Error $e) {
    echo "FAIL: Inverse trigonometric functions failed: " . $e->getMessage() . "\n";
}

// Test 2: Hyperbolic functions
try {
    $tensor = new ONNX\Tensor\Transient([3], [0.0, 1.0, 2.0], ONNX\Tensor::FLOAT);
    
    $result = ONNX\Math\sinh($tensor);
    echo "PASS: sinh function works\n";
    
    $result = ONNX\Math\cosh($tensor);
    echo "PASS: cosh function works\n";
    
    $result = ONNX\Math\tanh($tensor);
    echo "PASS: tanh function works\n";
} catch (Error $e) {
    echo "FAIL: Hyperbolic functions failed: " . $e->getMessage() . "\n";
}

// Test 3: Exponential and logarithmic functions
try {
    $tensor = new ONNX\Tensor\Transient([3], [1.0, 2.0, 4.0], ONNX\Tensor::FLOAT);
    
    $result = ONNX\Math\exp($tensor);
    echo "PASS: exp function works\n";
    
    $result = ONNX\Math\exp2($tensor);
    echo "PASS: exp2 function works\n";
    
    $result = ONNX\Math\log($tensor);
    echo "PASS: log function works\n";
    
    $result = ONNX\Math\log2($tensor);
    echo "PASS: log2 function works\n";
    
    $result = ONNX\Math\log10($tensor);
    echo "PASS: log10 function works\n";
} catch (Error $e) {
    echo "FAIL: Exponential/logarithmic functions failed: " . $e->getMessage() . "\n";
}

// Test 4: Rounding and utility functions
try {
    $tensor = new ONNX\Tensor\Transient([6], [2.3, -2.7, 8.9, -8.1, 0.0, 5.5], ONNX\Tensor::FLOAT);
    
    $result = ONNX\Math\ceil($tensor);
    echo "PASS: ceil function works\n";
    
    $result = ONNX\Math\floor($tensor);
    echo "PASS: floor function works\n";
    
    $result = ONNX\Math\round($tensor);
    echo "PASS: round function works\n";
    
    $result = ONNX\Math\abs($tensor);
    echo "PASS: abs function works\n";
    
    $result = ONNX\Math\sign($tensor);
    echo "PASS: sign function works\n";
} catch (Error $e) {
    echo "FAIL: Rounding/utility functions failed: " . $e->getMessage() . "\n";
}

// Test 5: Cube root function
try {
    $tensor = new ONNX\Tensor\Transient([4], [8.0, 27.0, 64.0, -8.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\cbrt($tensor);
    echo "PASS: cbrt function works\n";
} catch (Error $e) {
    echo "FAIL: cbrt function failed: " . $e->getMessage() . "\n";
}

// Test 6: pow function (element-wise)
try {
    $base = new ONNX\Tensor\Transient([3], [2.0, 3.0, 4.0], ONNX\Tensor::FLOAT);
    $exponent = new ONNX\Tensor\Transient([3], [2.0, 3.0, 0.5], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\pow($base, $exponent);
    echo "PASS: pow function (element-wise) works\n";
} catch (Error $e) {
    echo "FAIL: pow function failed: " . $e->getMessage() . "\n";
}

// Test 7: mod function (element-wise)
try {
    $base = new ONNX\Tensor\Transient([3], [2.0, 3.0, 4.0], ONNX\Tensor::FLOAT);
    $mod = new ONNX\Tensor\Transient([3], [2.0, 3.0, 0.5], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\mod($base, $mod);
    echo "PASS: mod function (element-wise) works\n";
} catch (Error $e) {
    echo "FAIL: mod function failed: " . $e->getMessage() . "\n";
}

// Test 8: pow function (scalar)
try {
    $tensor = new ONNX\Tensor\Transient([3], [2.0, 3.0, 4.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\pow($tensor, 2.0);
    echo "PASS: pow function (scalar) works\n";
} catch (Error $e) {
    echo "FAIL: pow scalar function failed: " . $e->getMessage() . "\n";
}

// Test 9: mod function (scalar)
try {
    $tensor = new ONNX\Tensor\Transient([3], [2.0, 3.0, 4.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\mod($tensor, 2.0);
    echo "PASS: mod function (scalar) works\n";
} catch (Error $e) {
    echo "FAIL: mod scalar function failed: " . $e->getMessage() . "\n";
}

// Test 10: Error handling - sqrt of negative numbers
try {
    $tensor = new ONNX\Tensor\Transient([3], [-1.0, 4.0, -9.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($tensor);
    $data = $result->getData();
    // Should produce NaN for negative values
    echo "INFO: sqrt of negative numbers produces: [" . implode(", ", array_map(function($x) { return is_nan($x) ? "NaN" : $x; }, $data)) . "]\n";
} catch (Error $e) {
    echo "INFO: sqrt of negative numbers throws: " . get_class($e) . "\n";
}

// Test 11: Error handling - log of zero and negative numbers
try {
    $tensor = new ONNX\Tensor\Transient([3], [0.0, 1.0, -1.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\log($tensor);
    $data = $result->getData();
    // Should produce -INF for 0, NaN for negative
    echo "INFO: log of [0, 1, -1] produces: [" . implode(", ", array_map(function($x) { 
        if (is_infinite($x)) return $x < 0 ? "-INF" : "INF";
        if (is_nan($x)) return "NaN";
        return $x;
    }, $data)) . "]\n";
} catch (Error $e) {
    echo "INFO: log of zero/negative throws: " . get_class($e) . "\n";
}

// Test 12: Edge cases - very large and very small numbers
try {
    $tensor = new ONNX\Tensor\Transient([4], [1e10, 1e-10, 0.0, 1.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\exp($tensor);
    echo "PASS: exp with extreme values works\n";
    
    $result = ONNX\Math\sqrt($tensor);
    echo "PASS: sqrt with extreme values works\n";
} catch (Error $e) {
    echo "FAIL: Extreme value handling failed: " . $e->getMessage() . "\n";
}

// Test 13: Trigonometric functions with extreme values
try {
    $tensor = new ONNX\Tensor\Transient([4], [0.0, 1.5707963267948966, 3.141592653589793, 6.283185307179586], ONNX\Tensor::FLOAT);
    
    $result = ONNX\Math\sin($tensor);
    $data = $result->getData();
    // sin(0)=0, sin(π/2)=1, sin(π)=0, sin(2π)=0
    echo "INFO: sin([0, π/2, π, 2π]) ≈ [" . implode(", ", array_map(function($x) { return round($x, 6); }, $data)) . "]\n";
    
    $result = ONNX\Math\cos($tensor);
    $data = $result->getData();
    // cos(0)=1, cos(π/2)=0, cos(π)=-1, cos(2π)=1
    echo "INFO: cos([0, π/2, π, 2π]) ≈ [" . implode(", ", array_map(function($x) { return round($x, 6); }, $data)) . "]\n";
} catch (Error $e) {
    echo "FAIL: Trigonometric extreme values failed: " . $e->getMessage() . "\n";
}

// Test 14: Chained operations
try {
    $tensor = new ONNX\Tensor\Transient([3], [4.0, 9.0, 16.0], ONNX\Tensor::FLOAT);
    $sqrt_result = ONNX\Math\sqrt($tensor);
    $squared_result = ONNX\Math\pow($sqrt_result, 2.0);
    $data = $squared_result->getData();
    // Should get back close to original values
    if (abs($data[0] - 4.0) < 0.001 && abs($data[1] - 9.0) < 0.001 && abs($data[2] - 16.0) < 0.001) {
        echo "PASS: Chained operations (sqrt then square) work correctly\n";
    } else {
        echo "FAIL: Chained operations: expected [4, 9, 16], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: Chained operations failed: " . $e->getMessage() . "\n";
}

echo "Unary math functions comprehensive testing completed!\n";
?>
--EXPECT--
=== Testing Unary Math Functions - Comprehensive Coverage ===
PASS: asin function works
PASS: acos function works
PASS: atan function works
PASS: sinh function works
PASS: cosh function works
PASS: tanh function works
PASS: exp function works
PASS: exp2 function works
PASS: log function works
PASS: log2 function works
PASS: log10 function works
PASS: ceil function works
PASS: floor function works
PASS: round function works
PASS: abs function works
PASS: sign function works
PASS: cbrt function works
PASS: pow function (element-wise) works
PASS: mod function (element-wise) works
PASS: pow function (scalar) works
PASS: mod function (scalar) works
INFO: sqrt of negative numbers produces: [NaN, 2, NaN]
INFO: log of [0, 1, -1] produces: [-INF, 0, NaN]
PASS: exp with extreme values works
PASS: sqrt with extreme values works
INFO: sin([0, π/2, π, 2π]) ≈ [0, 1, -0, 0]
INFO: cos([0, π/2, π, 2π]) ≈ [1, -0, -1, 1]
PASS: Chained operations (sqrt then square) work correctly
Unary math functions comprehensive testing completed!
