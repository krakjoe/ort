--TEST--
Check ONNX\Math deep testing - error handling and edge cases
--EXTENSIONS--
ort
--FILE--
<?php
// Deep testing for math functions - error handling and edge cases

echo "=== Testing Error Handling ===\n";

// Test 1: Invalid tensor types
try {
    $tensor = new ONNX\Tensor\Transient([2], [1.0, 2.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\add($tensor, "invalid_scalar");
    echo "ERROR: Should have thrown exception for invalid scalar\n";
} catch (ONNX\Status\Math\InvalidType $e) {
    echo "PASS: add() correctly rejects invalid scalar type\n";
}

// Test 2: Divide by zero (should work and return INF)
try {
    $tensor = new ONNX\Tensor\Transient([2], [1.0, 2.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\divide($tensor, 0.0);
    $data = $result->getData();
    if (is_infinite($data[0]) && is_infinite($data[1])) {
        echo "PASS: divide() by zero returns INF values\n";
    } else {
        echo "INFO: divide() by zero result: " . implode(", ", $data) . "\n";
    }
} catch (Error $e) {
    echo "INFO: divide() throws exception for zero: " . get_class($e) . "\n";
}

// Test 3: Very small tensors
try {
    $tensor = new ONNX\Tensor\Transient([1], [42.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($tensor);
    echo "PASS: sqrt() handles 1-element tensor\n";
} catch (Error $e) {
    echo "ERROR: sqrt() failed on 1-element tensor: " . $e->getMessage() . "\n";
}

// Test 5: Negative values in sqrt (should work and return NaN)
try {
    $tensor = new ONNX\Tensor\Transient([2], [-1.0, 4.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($tensor);
    $data = $result->getData();
    if (is_nan($data[0]) && $data[1] == 2.0) {
        echo "PASS: sqrt() handles negative values correctly (NaN for -1, 2 for 4)\n";
    } else {
        echo "INFO: sqrt() with negative result: " . implode(", ", $data) . "\n";
    }
} catch (Error $e) {
    echo "INFO: sqrt() throws exception for negative: " . get_class($e) . "\n";
}

echo "\n=== Testing Edge Cases ===\n";

// Test 6: Very large numbers
try {
    $tensor = new ONNX\Tensor\Transient([2], [1e10, 1e10], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\multiply($tensor, 1e10);
    echo "PASS: multiply() handles large numbers\n";
} catch (Error $e) {
    echo "ERROR: multiply() failed on large numbers: " . $e->getMessage() . "\n";
}

// Test 7: Very small numbers
try {
    $tensor = new ONNX\Tensor\Transient([2], [1e-10, 1e-10], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\add($tensor, 1e-10);
    echo "PASS: add() handles small numbers\n";
} catch (Error $e) {
    echo "ERROR: add() failed on small numbers: " . $e->getMessage() . "\n";
}

// Test 8: Matrix multiplication dimension mismatch
try {
    $matrixA = new ONNX\Tensor\Transient([2, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ONNX\Tensor::FLOAT);
    $matrixB = new ONNX\Tensor\Transient([2, 2], [[1.0, 2.0], [3.0, 4.0]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\matmul($matrixA, $matrixB);
    echo "ERROR: matmul() should reject dimension mismatch\n";
} catch (ONNX\Status\Math\Error $e) {
    echo "PASS: matmul() correctly rejects dimension mismatch\n";
}

// Test 10: pow with edge cases
try {
    $tensor = new ONNX\Tensor\Transient([3], [0.0, 1.0, -1.0], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\pow($tensor, 2.0);
    echo "PASS: pow() handles edge values (0, 1, -1)\n";
} catch (Error $e) {
    echo "ERROR: pow() failed on edge values: " . $e->getMessage() . "\n";
}

echo "\n=== Testing Data Type Conversions ===\n";

// Test 11: Mixed integer/float operations
try {
    $tensor = new ONNX\Tensor\Transient([2], [1, 2], ONNX\Tensor::INT32);
    $result = ONNX\Math\add($tensor, 3.5);
    echo "PASS: add() handles INT32 tensor with float scalar\n";
} catch (Error $e) {
    echo "ERROR: add() failed on mixed types: " . $e->getMessage() . "\n";
}

// Test 12: Boolean tensor operations
try {
    $tensor = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
    $result = ONNX\Math\add($tensor, 1);
    echo "PASS: add() handles boolean tensors\n";
} catch (Error $e) {
    echo "INFO: add() rejects boolean tensors: " . get_class($e) . "\n";
}

echo "\n=== Testing Return Value Consistency ===\n";

// Test 13: Verify all functions return Transient tensors
$functions = ['add', 'multiply', 'sqrt'];
foreach ($functions as $func) {
    try {
        $tensor = new ONNX\Tensor\Transient([2], [1.0, 2.0], ONNX\Tensor::FLOAT);
        if ($func === 'add' || $func === 'multiply') {
            $result = call_user_func("ONNX\\Math\\$func", $tensor, 1.0);
        } else {
            $result = call_user_func("ONNX\\Math\\$func", $tensor);
        }
        
        $class = get_class($result);
        if ($class === 'ONNX\\Tensor\\Transient') {
            echo "PASS: $func() returns Transient tensor\n";
        } else {
            echo "ERROR: $func() returns $class instead of Transient\n";
        }        } catch (Error $e) {
            echo "ERROR: $func() failed: " . $e->getMessage() . "\n";
    }
}

echo "\nDeep testing completed.\n";
?>
--EXPECT--
=== Testing Error Handling ===
PASS: add() correctly rejects invalid scalar type
PASS: divide() by zero returns INF values
PASS: sqrt() handles 1-element tensor
PASS: sqrt() handles negative values correctly (NaN for -1, 2 for 4)

=== Testing Edge Cases ===
PASS: multiply() handles large numbers
PASS: add() handles small numbers
PASS: matmul() correctly rejects dimension mismatch
PASS: pow() handles edge values (0, 1, -1)

=== Testing Data Type Conversions ===
PASS: add() handles INT32 tensor with float scalar
PASS: add() handles boolean tensors

=== Testing Return Value Consistency ===
PASS: add() returns Transient tensor
PASS: multiply() returns Transient tensor
PASS: sqrt() returns Transient tensor

Deep testing completed.
