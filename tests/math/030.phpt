--TEST--
Test integer division operations - comprehensive coverage
--EXTENSIONS--
ort
--FILE--
<?php
echo "=== Testing Integer Division Operations ===\n";

// Test 1: INT8 division
try {
    $tensor_a = new ONNX\Tensor\Transient([3], [24, 16, 8], ONNX\Tensor::INT8);
    $tensor_b = new ONNX\Tensor\Transient([3], [3, 2, 4], ONNX\Tensor::INT8);
    $result = ONNX\Math\divide($tensor_a, $tensor_b);
    $data = $result->getData();
    if ($data[0] == 8 && $data[1] == 8 && $data[2] == 2) {
        echo "PASS: INT8 division works\n";
    } else {
        echo "FAIL: INT8 division: expected [8, 8, 2], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: INT8 division failed: " . $e->getMessage() . "\n";
}

// Test 2: INT16 division
try {
    $tensor_a = new ONNX\Tensor\Transient([3], [1000, 2000, 3000], ONNX\Tensor::INT16);
    $tensor_b = new ONNX\Tensor\Transient([3], [10, 20, 30], ONNX\Tensor::INT16);
    $result = ONNX\Math\divide($tensor_a, $tensor_b);
    $data = $result->getData();
    if ($data[0] == 100 && $data[1] == 100 && $data[2] == 100) {
        echo "PASS: INT16 division works\n";
    } else {
        echo "FAIL: INT16 division: expected [100, 100, 100], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: INT16 division failed: " . $e->getMessage() . "\n";
}

// Test 3: INT32 division
try {
    $tensor_a = new ONNX\Tensor\Transient([3], [100000, 200000, 300000], ONNX\Tensor::INT32);
    $tensor_b = new ONNX\Tensor\Transient([3], [1000, 2000, 3000], ONNX\Tensor::INT32);
    $result = ONNX\Math\divide($tensor_a, $tensor_b);
    $data = $result->getData();
    if ($data[0] == 100 && $data[1] == 100 && $data[2] == 100) {
        echo "PASS: INT32 division works\n";
    } else {
        echo "FAIL: INT32 division: expected [100, 100, 100], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: INT32 division failed: " . $e->getMessage() . "\n";
}

// Test 4: INT64 division
try {
    $tensor_a = new ONNX\Tensor\Transient([3], [1000000000, 2000000000, 3000000000], ONNX\Tensor::INT64);
    $tensor_b = new ONNX\Tensor\Transient([3], [10000000, 20000000, 30000000], ONNX\Tensor::INT64);
    $result = ONNX\Math\divide($tensor_a, $tensor_b);
    $data = $result->getData();
    if ($data[0] == 100 && $data[1] == 100 && $data[2] == 100) {
        echo "PASS: INT64 division works\n";
    } else {
        echo "FAIL: INT64 division: expected [100, 100, 100], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: INT64 division failed: " . $e->getMessage() . "\n";
}

// Test 5: UINT8 division
try {
    $tensor_a = new ONNX\Tensor\Transient([3], [240, 160, 80], ONNX\Tensor::UINT8);
    $tensor_b = new ONNX\Tensor\Transient([3], [3, 2, 4], ONNX\Tensor::UINT8);
    $result = ONNX\Math\divide($tensor_a, $tensor_b);
    $data = $result->getData();
    if ($data[0] == 80 && $data[1] == 80 && $data[2] == 20) {
        echo "PASS: UINT8 division works\n";
    } else {
        echo "FAIL: UINT8 division: expected [80, 80, 20], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: UINT8 division failed: " . $e->getMessage() . "\n";
}

// Test 6: UINT16 division
try {
    $tensor_a = new ONNX\Tensor\Transient([3], [10000, 20000, 30000], ONNX\Tensor::UINT16);
    $tensor_b = new ONNX\Tensor\Transient([3], [100, 200, 300], ONNX\Tensor::UINT16);
    $result = ONNX\Math\divide($tensor_a, $tensor_b);
    $data = $result->getData();
    if ($data[0] == 100 && $data[1] == 100 && $data[2] == 100) {
        echo "PASS: UINT16 division works\n";
    } else {
        echo "FAIL: UINT16 division: expected [100, 100, 100], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: UINT16 division failed: " . $e->getMessage() . "\n";
}

// Test 7: UINT32 division
try {
    $tensor_a = new ONNX\Tensor\Transient([3], [1000000, 2000000, 3000000], ONNX\Tensor::UINT32);
    $tensor_b = new ONNX\Tensor\Transient([3], [10000, 20000, 30000], ONNX\Tensor::UINT32);
    $result = ONNX\Math\divide($tensor_a, $tensor_b);
    $data = $result->getData();
    if ($data[0] == 100 && $data[1] == 100 && $data[2] == 100) {
        echo "PASS: UINT32 division works\n";
    } else {
        echo "FAIL: UINT32 division: expected [100, 100, 100], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: UINT32 division failed: " . $e->getMessage() . "\n";
}

echo "Integer division operations testing completed!\n";
?>
--EXPECT--
=== Testing Integer Division Operations ===
PASS: INT8 division works
PASS: INT16 division works
PASS: INT32 division works
PASS: INT64 division works
PASS: UINT8 division works
PASS: UINT16 division works
PASS: UINT32 division works
Integer division operations testing completed!
