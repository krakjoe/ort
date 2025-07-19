--TEST--
Test individual math function - sqrt (focused unit test)
--EXTENSIONS--
ort
--FILE--
<?php
// Focused unit test for sqrt function only

echo "=== Testing SQRT Function ===\n";

// Test 1: Basic functionality
try {
    $tensor = new ORT\Tensor\Transient([2], [4.0, 9.0], ORT\Tensor::FLOAT);
    $result = ORT\Math\sqrt($tensor);
    $data = $result->getData();
    if (abs($data[0] - 2.0) < 0.001 && abs($data[1] - 3.0) < 0.001) {
        echo "PASS: sqrt([4, 9]) = [2, 3]\n";
    } else {
        echo "FAIL: sqrt([4, 9]) = [" . $data[0] . ", " . $data[1] . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: Basic sqrt failed: " . $e->getMessage() . "\n";
}

// Test 2: Edge case - zero
try {
    $tensor = new ORT\Tensor\Transient([1], [0.0], ORT\Tensor::FLOAT);
    $result = ORT\Math\sqrt($tensor);
    $data = $result->getData();
    if (abs($data[0] - 0.0) < 0.001) {
        echo "PASS: sqrt(0) = 0\n";
    } else {
        echo "FAIL: sqrt(0) = " . $data[0] . "\n";
    }
} catch (Error $e) {
    echo "FAIL: sqrt(0) failed: " . $e->getMessage() . "\n";
}

// Test 3: Edge case - one
try {
    $tensor = new ORT\Tensor\Transient([1], [1.0], ORT\Tensor::FLOAT);
    $result = ORT\Math\sqrt($tensor);
    $data = $result->getData();
    if (abs($data[0] - 1.0) < 0.001) {
        echo "PASS: sqrt(1) = 1\n";
    } else {
        echo "FAIL: sqrt(1) = " . $data[0] . "\n";
    }
} catch (Error $e) {
    echo "FAIL: sqrt(1) failed: " . $e->getMessage() . "\n";
}

// Test 4: Edge case - negative number (should return NaN)
try {
    $tensor = new ORT\Tensor\Transient([1], [-1.0], ORT\Tensor::FLOAT);
    $result = ORT\Math\sqrt($tensor);
    $data = $result->getData();
    if (is_nan($data[0])) {
        echo "PASS: sqrt(-1) = NaN\n";
    } else {
        echo "INFO: sqrt(-1) = " . $data[0] . " (expected NaN)\n";
    }
} catch (Error $e) {
    echo "INFO: sqrt(-1) throws exception: " . get_class($e) . "\n";
}

// Test 5: Different data types
try {
    $tensor_double = new ORT\Tensor\Transient([2], [16.0, 25.0], ORT\Tensor::DOUBLE);
    $result = ORT\Math\sqrt($tensor_double);
    $data = $result->getData();
    if (abs($data[0] - 4.0) < 0.001 && abs($data[1] - 5.0) < 0.001) {
        echo "PASS: sqrt(DOUBLE) works correctly\n";
    } else {
        echo "FAIL: sqrt(DOUBLE) = [" . $data[0] . ", " . $data[1] . "]\n";
    }
} catch (Error $e) {
    echo "INFO: sqrt(DOUBLE) not supported: " . get_class($e) . "\n";
}

// Test 6: Large numbers
try {
    $tensor = new ORT\Tensor\Transient([2], [10000.0, 1000000.0], ORT\Tensor::FLOAT);
    $result = ORT\Math\sqrt($tensor);
    $data = $result->getData();
    if (abs($data[0] - 100.0) < 0.001 && abs($data[1] - 1000.0) < 0.001) {
        echo "PASS: sqrt(large numbers) works correctly\n";
    } else {
        echo "FAIL: sqrt(large) = [" . $data[0] . ", " . $data[1] . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: sqrt(large numbers) failed: " . $e->getMessage() . "\n";
}

// Test 7: Very small numbers
try {
    $tensor = new ORT\Tensor\Transient([2], [0.01, 0.0001], ORT\Tensor::FLOAT);
    $result = ORT\Math\sqrt($tensor);
    $data = $result->getData();
    if (abs($data[0] - 0.1) < 0.001 && abs($data[1] - 0.01) < 0.001) {
        echo "PASS: sqrt(small numbers) works correctly\n";
    } else {
        echo "FAIL: sqrt(small) = [" . $data[0] . ", " . $data[1] . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: sqrt(small numbers) failed: " . $e->getMessage() . "\n";
}

// Test 8: Return type validation
try {
    $tensor = new ORT\Tensor\Transient([1], [4.0], ORT\Tensor::FLOAT);
    $result = ORT\Math\sqrt($tensor);
    if (get_class($result) === 'ORT\\Tensor\\Transient') {
        echo "PASS: sqrt() returns Transient tensor\n";
    } else {
        echo "FAIL: sqrt() returns " . get_class($result) . "\n";
    }
} catch (Error $e) {
    echo "FAIL: Return type test failed: " . $e->getMessage() . "\n";
}

echo "SQRT function tests completed.\n";
?>
--EXPECT--
=== Testing SQRT Function ===
PASS: sqrt([4, 9]) = [2, 3]
PASS: sqrt(0) = 0
PASS: sqrt(1) = 1
PASS: sqrt(-1) = NaN
PASS: sqrt(DOUBLE) works correctly
PASS: sqrt(large numbers) works correctly
PASS: sqrt(small numbers) works correctly
PASS: sqrt() returns Transient tensor
SQRT function tests completed.
