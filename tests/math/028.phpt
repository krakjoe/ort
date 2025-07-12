--TEST--
Test unary math functions - comprehensive coverage  
--EXTENSIONS--
ort
--FILE--
<?php
echo "=== Testing Unary Math Functions - Comprehensive Coverage ===\n";

// Test 4: Rounding and utility functions
foreach ([
    "\ONNX\Tensor::DOUBLE",
    "\ONNX\Tensor::FLOAT"
] as $type) {
    try {
        $tensor = \ONNX\Tensor\Transient::from(
            [2.3, -2.7, 8.9, -8.1, 0.0, 5.5],
            \constant($type));
        
        $result = ONNX\Math\ceil($tensor);
        echo "PASS: ceil $type function works\n";
        
        $result = ONNX\Math\floor($tensor);
        echo "PASS: floor $type function works\n";
        
        $result = ONNX\Math\round($tensor);
        echo "PASS: round $type function works\n";
        
        $result = ONNX\Math\abs($tensor);
        echo "PASS: abs $type function works\n";
        
        $result = ONNX\Math\sign($tensor);
        echo "PASS: sign $type function works\n";
    } catch (Error $e) {
        echo "FAIL: Rounding/utility functions failed: " . $e->getMessage() . "\n";
    }
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

// Test 15: mod vectors
try {
    foreach ([
        "\ONNX\Tensor::INT8", 
        "\ONNX\Tensor::INT16", 
        "\ONNX\Tensor::INT32",
        "\ONNX\Tensor::INT64",
        "\ONNX\Tensor::UINT8",
        "\ONNX\Tensor::UINT16",
        "\ONNX\Tensor::UINT32",
        "\ONNX\Tensor::FLOAT",
        "\ONNX\Tensor::DOUBLE"] as $type) {
        $number = \ONNX\Tensor\Transient::from(
            [20], \constant($type));
        $eleven = \ONNX\Tensor\Transient::from(
            [11], \constant($type));

        $result = \ONNX\Math\mod($number, $eleven);

        $data   = $result->getData();
        if ($data[0] == 9) {
            echo "PASS: mod $type passed\n";
        } else {
            echo "FAIL: mod $type failed\n";
        }
    }
} catch (Error $e) {
    echo "FAIL: mod with $type failed\n";
    echo (string) $e;
}

try {
    foreach ([
        "\ONNX\Tensor::INT8", 
        "\ONNX\Tensor::INT16", 
        "\ONNX\Tensor::INT32",
        "\ONNX\Tensor::INT64",
        "\ONNX\Tensor::UINT8",
        "\ONNX\Tensor::UINT16",
        "\ONNX\Tensor::UINT32",
        "\ONNX\Tensor::FLOAT",
        "\ONNX\Tensor::DOUBLE"] as $type) {
        $number = \ONNX\Tensor\Transient::from(
            [12], \constant($type));

        $result = \ONNX\Math\mod($number, 10);

        $data   = $result->getData();
        if ($data[0] == 2) {
            echo "PASS: mod scalar $type passed\n";
        } else {
            echo "FAIL: mod scalar $type failed\n";
        }
    }
} catch (Error $e) {
    echo "FAIL: mod scalar with $type failed\n";
    echo (string) $e;
}

echo "Unary math functions comprehensive testing completed!\n";
?>
--EXPECT--
=== Testing Unary Math Functions - Comprehensive Coverage ===
PASS: ceil \ONNX\Tensor::DOUBLE function works
PASS: floor \ONNX\Tensor::DOUBLE function works
PASS: round \ONNX\Tensor::DOUBLE function works
PASS: abs \ONNX\Tensor::DOUBLE function works
PASS: sign \ONNX\Tensor::DOUBLE function works
PASS: ceil \ONNX\Tensor::FLOAT function works
PASS: floor \ONNX\Tensor::FLOAT function works
PASS: round \ONNX\Tensor::FLOAT function works
PASS: abs \ONNX\Tensor::FLOAT function works
PASS: sign \ONNX\Tensor::FLOAT function works
PASS: pow function (element-wise) works
PASS: mod function (element-wise) works
PASS: pow function (scalar) works
PASS: mod function (scalar) works
PASS: mod \ONNX\Tensor::INT8 passed
PASS: mod \ONNX\Tensor::INT16 passed
PASS: mod \ONNX\Tensor::INT32 passed
PASS: mod \ONNX\Tensor::INT64 passed
PASS: mod \ONNX\Tensor::UINT8 passed
PASS: mod \ONNX\Tensor::UINT16 passed
PASS: mod \ONNX\Tensor::UINT32 passed
PASS: mod \ONNX\Tensor::FLOAT passed
PASS: mod \ONNX\Tensor::DOUBLE passed
PASS: mod scalar \ONNX\Tensor::INT8 passed
PASS: mod scalar \ONNX\Tensor::INT16 passed
PASS: mod scalar \ONNX\Tensor::INT32 passed
PASS: mod scalar \ONNX\Tensor::INT64 passed
PASS: mod scalar \ONNX\Tensor::UINT8 passed
PASS: mod scalar \ONNX\Tensor::UINT16 passed
PASS: mod scalar \ONNX\Tensor::UINT32 passed
PASS: mod scalar \ONNX\Tensor::FLOAT passed
PASS: mod scalar \ONNX\Tensor::DOUBLE passed
Unary math functions comprehensive testing completed!
