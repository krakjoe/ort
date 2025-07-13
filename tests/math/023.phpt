--TEST--
Test boolean tensor operations - comprehensive coverage
--EXTENSIONS--
ort
--FILE--
<?php
echo "=== Testing Boolean Tensor Operations ===\n";

// Test 1: Boolean tensor addition (logical OR)
try {
    $tensor_a = new ONNX\Tensor\Transient([2, 2], [[true, false], [false, true]], ONNX\Tensor::BOOL);
    $tensor_b = new ONNX\Tensor\Transient([2, 2], [[false, true], [true, false]], ONNX\Tensor::BOOL);
    $result = ONNX\Math\add($tensor_a, $tensor_b);
    $data = $result->getData();
    // OR: [true||false, false||true], [false||true, true||false] = [true, true], [true, true]
    // getData returns structured array, so we need to flatten it or access correctly
    if ($data[0][0] && $data[0][1] && $data[1][0] && $data[1][1]) {
        echo "PASS: Boolean addition (OR) works correctly\n";
    } else {
        echo "FAIL: Boolean addition result: " . json_encode($data) . "\n";
    }
} catch (Error $e) {
    echo "FAIL: Boolean addition failed: " . $e->getMessage() . "\n";
}

// Test 3: Boolean tensor multiplication (logical AND)
try {
    $tensor_a = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
    $tensor_b = new ONNX\Tensor\Transient([2], [true, true], ONNX\Tensor::BOOL);
    $result = ONNX\Math\multiply($tensor_a, $tensor_b);
    $data = $result->getData();
    // AND: [true && true, false && true] = [true, false]
    if ($data[0] === true && $data[1] === false) {
        echo "PASS: Boolean multiplication (AND) works correctly\n";
    } else {
        echo "FAIL: Boolean multiplication result: " . json_encode($data) . "\n";
    }
} catch (Error $e) {
    echo "FAIL: Boolean multiplication failed: " . $e->getMessage() . "\n";
}

// Test 5: Boolean scalar operations - addition (using 0/1 for false/true)
try {
    $tensor = new ONNX\Tensor\Transient([3], [true, false, true], ONNX\Tensor::BOOL);
    $result = ONNX\Math\add($tensor, 0); // 0 = false
    $data = $result->getData();
    // OR with false: [true||false, false||false, true||false] = [true, false, true]
    if ($data[0] === true && $data[1] === false && $data[2] === true) {
        echo "PASS: Boolean scalar addition works correctly\n";
    } else {
        echo "FAIL: Boolean scalar addition result: " . json_encode($data) . "\n";
    }
} catch (Error $e) {
    echo "FAIL: Boolean scalar addition failed: " . $e->getMessage() . "\n";
}

// Test 7: Boolean scalar multiplication
try {
    $tensor = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
    $result = ONNX\Math\multiply($tensor, 1); // 1 = true
    $data = $result->getData();
    // AND: [true && true, false && true] = [true, false]
    if ($data[0] === true && $data[1] === false) {
        echo "PASS: Boolean scalar multiplication works correctly\n";
    } else {
        echo "FAIL: Boolean scalar multiplication result: " . json_encode($data) . "\n";
    }
} catch (Error $e) {
    echo "FAIL: Boolean scalar multiplication failed: " . $e->getMessage() . "\n";
}

echo "Boolean tensor operation tests completed!\n";
?>
--EXPECT--
=== Testing Boolean Tensor Operations ===
PASS: Boolean addition (OR) works correctly
PASS: Boolean multiplication (AND) works correctly
PASS: Boolean scalar addition works correctly
PASS: Boolean scalar multiplication works correctly
Boolean tensor operation tests completed!
