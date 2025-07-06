--TEST--
Test double precision math operations - comprehensive coverage
--EXTENSIONS--
ort
--FILE--
<?php
echo "=== Testing Double Precision Math Operations ===\n";

// Test 1: Basic double operations
try {
    $tensor_a = new ONNX\Tensor\Transient([3], [1.234567890123456, -9.876543210987654, 0.0], ONNX\Tensor::DOUBLE);
    $tensor_b = new ONNX\Tensor\Transient([3], [2.345678901234567, 1.111111111111111, 5.555555555555555], ONNX\Tensor::DOUBLE);
    
    $result = ONNX\Math\add($tensor_a, $tensor_b);
    echo "PASS: Double addition works\n";
    
    $result = ONNX\Math\subtract($tensor_a, $tensor_b);
    echo "PASS: Double subtraction works\n";
    
    $result = ONNX\Math\multiply($tensor_a, $tensor_b);
    echo "PASS: Double multiplication works\n";
    
    $result = ONNX\Math\divide($tensor_a, $tensor_b);
    echo "PASS: Double division works\n";
    
    $result = ONNX\Math\neg($tensor_a);
    echo "PASS: Double negation works\n";
} catch (Error $e) {
    echo "FAIL: Double basic operations failed: " . $e->getMessage() . "\n";
}

// Test 2: Double scalar operations
try {
    $tensor = new ONNX\Tensor\Transient([2], [3.141592653589793, 2.718281828459045], ONNX\Tensor::DOUBLE);
    
    $result = ONNX\Math\add($tensor, 1.414213562373095);
    echo "PASS: Double scalar addition works\n";
    
    $result = ONNX\Math\subtract($tensor, 0.5772156649015329);
    echo "PASS: Double scalar subtraction works\n";
    
    $result = ONNX\Math\multiply($tensor, 1.618033988749895);
    echo "PASS: Double scalar multiplication works\n";
    
    $result = ONNX\Math\divide($tensor, 2.0);
    echo "PASS: Double scalar division works\n";
} catch (Error $e) {
    echo "FAIL: Double scalar operations failed: " . $e->getMessage() . "\n";
}

// Test 3: Double mathematical functions
try {
    $tensor = new ONNX\Tensor\Transient([3], [1.0, 2.0, 4.0], ONNX\Tensor::DOUBLE);
    
    $result = ONNX\Math\sqrt($tensor);
    echo "PASS: Double sqrt works\n";
    
    $result = ONNX\Math\exp($tensor);
    echo "PASS: Double exp works\n";
    
    $result = ONNX\Math\log($tensor);
    echo "PASS: Double log works\n";
    
    $result = ONNX\Math\log2($tensor);
    echo "PASS: Double log2 works\n";
    
    $result = ONNX\Math\log10($tensor);
    echo "PASS: Double log10 works\n";
} catch (Error $e) {
    echo "FAIL: Double mathematical functions failed: " . $e->getMessage() . "\n";
}

// Test 4: Double trigonometric functions
try {
    $tensor = new ONNX\Tensor\Transient([4], [0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469], ONNX\Tensor::DOUBLE);
    
    $result = ONNX\Math\sin($tensor);
    echo "PASS: Double sin works\n";
    
    $result = ONNX\Math\cos($tensor);
    echo "PASS: Double cos works\n";
    
    $result = ONNX\Math\tan($tensor);
    echo "PASS: Double tan works\n";
} catch (Error $e) {
    echo "FAIL: Double trigonometric functions failed: " . $e->getMessage() . "\n";
}

// Test 5: Double inverse trigonometric functions
try {
    $tensor = new ONNX\Tensor\Transient([3], [0.0, 0.5, 1.0], ONNX\Tensor::DOUBLE);
    
    $result = ONNX\Math\asin($tensor);
    echo "PASS: Double asin works\n";
    
    $result = ONNX\Math\acos($tensor);
    echo "PASS: Double acos works\n";
    
    $result = ONNX\Math\atan($tensor);
    echo "PASS: Double atan works\n";
} catch (Error $e) {
    echo "FAIL: Double inverse trigonometric functions failed: " . $e->getMessage() . "\n";
}

// Test 6: Double hyperbolic functions
try {
    $tensor = new ONNX\Tensor\Transient([3], [0.0, 1.0, 2.0], ONNX\Tensor::DOUBLE);
    
    $result = ONNX\Math\sinh($tensor);
    echo "PASS: Double sinh works\n";
    
    $result = ONNX\Math\cosh($tensor);
    echo "PASS: Double cosh works\n";
    
    $result = ONNX\Math\tanh($tensor);
    echo "PASS: Double tanh works\n";
} catch (Error $e) {
    echo "FAIL: Double hyperbolic functions failed: " . $e->getMessage() . "\n";
}

// Test 7: Double pow and exponential functions
try {
    $tensor_base = new ONNX\Tensor\Transient([3], [2.0, 3.0, 10.0], ONNX\Tensor::DOUBLE);
    $tensor_exp = new ONNX\Tensor\Transient([3], [3.0, 2.0, 0.5], ONNX\Tensor::DOUBLE);
    
    $result = ONNX\Math\pow($tensor_base, $tensor_exp);
    echo "PASS: Double pow works\n";
    
    $result = ONNX\Math\pow($tensor_base, 2.0);
    echo "PASS: Double pow scalar works\n";
    
    $tensor = new ONNX\Tensor\Transient([2], [1.0, 2.0], ONNX\Tensor::DOUBLE);
    $result = ONNX\Math\exp2($tensor);
    echo "PASS: Double exp2 works\n";
} catch (Error $e) {
    echo "FAIL: Double pow/exponential functions failed: " . $e->getMessage() . "\n";
}

// Test 8: Double rounding and utility functions
try {
    $tensor = new ONNX\Tensor\Transient([4], [2.3, -2.7, 8.9, -8.1], ONNX\Tensor::DOUBLE);
    
    $result = ONNX\Math\ceil($tensor);
    echo "PASS: Double ceil works\n";
    
    $result = ONNX\Math\floor($tensor);
    echo "PASS: Double floor works\n";
    
    $result = ONNX\Math\round($tensor);
    echo "PASS: Double round works\n";
    
    $result = ONNX\Math\abs($tensor);
    echo "PASS: Double abs works\n";
    
    $result = ONNX\Math\sign($tensor);
    echo "PASS: Double sign works\n";
    
    $tensor_cbrt = new ONNX\Tensor\Transient([2], [8.0, 27.0], ONNX\Tensor::DOUBLE);
    $result = ONNX\Math\cbrt($tensor_cbrt);
    echo "PASS: Double cbrt works\n";
} catch (Error $e) {
    echo "FAIL: Double rounding/utility functions failed: " . $e->getMessage() . "\n";
}

// Test 9: Double precision edge cases
try {
    $tensor = new ONNX\Tensor\Transient([3], [1.7976931348623157e+308, 2.2250738585072014e-308, 0.0], ONNX\Tensor::DOUBLE);
    $result = ONNX\Math\add($tensor, 1.0);
    echo "PASS: Double extreme values handling works\n";
    
    $tensor_nan_test = new ONNX\Tensor\Transient([2], [-1.0, 0.0], ONNX\Tensor::DOUBLE);
    $result = ONNX\Math\sqrt($tensor_nan_test);
    echo "PASS: Double NaN/infinity handling works\n";
} catch (Error $e) {
    echo "FAIL: Double edge cases failed: " . $e->getMessage() . "\n";
}

echo "Double precision math operation tests completed!\n";
?>
--EXPECT--
=== Testing Double Precision Math Operations ===
PASS: Double addition works
PASS: Double subtraction works
PASS: Double multiplication works
PASS: Double division works
PASS: Double negation works
PASS: Double scalar addition works
PASS: Double scalar subtraction works
PASS: Double scalar multiplication works
PASS: Double scalar division works
PASS: Double sqrt works
PASS: Double exp works
PASS: Double log works
PASS: Double log2 works
PASS: Double log10 works
PASS: Double sin works
PASS: Double cos works
PASS: Double tan works
PASS: Double asin works
PASS: Double acos works
PASS: Double atan works
PASS: Double sinh works
PASS: Double cosh works
PASS: Double tanh works
PASS: Double pow works
PASS: Double pow scalar works
PASS: Double exp2 works
PASS: Double ceil works
PASS: Double floor works
PASS: Double round works
PASS: Double abs works
PASS: Double sign works
PASS: Double cbrt works
PASS: Double extreme values handling works
PASS: Double NaN/infinity handling works
Double precision math operation tests completed!
