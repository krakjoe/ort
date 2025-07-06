--TEST--
Test scalar operations with integer types - comprehensive coverage
--EXTENSIONS--
ort
--FILE--
<?php
echo "=== Testing Scalar Operations with Integer Types ===\n";

// Test 1: INT8 scalar operations
try {
    $tensor = new ONNX\Tensor\Transient([3], [10, 20, 30], ONNX\Tensor::INT8);
    
    $result = ONNX\Math\add($tensor, 5);
    $data = $result->getData();
    if ($data[0] == 15 && $data[1] == 25 && $data[2] == 35) {
        echo "PASS: INT8 scalar addition works\n";
    } else {
        echo "FAIL: INT8 scalar addition: expected [15, 25, 35], got [" . implode(", ", $data) . "]\n";
    }
    
    $result = ONNX\Math\subtract($tensor, 5);
    $data = $result->getData();
    if ($data[0] == 5 && $data[1] == 15 && $data[2] == 25) {
        echo "PASS: INT8 scalar subtraction works\n";
    } else {
        echo "FAIL: INT8 scalar subtraction: expected [5, 15, 25], got [" . implode(", ", $data) . "]\n";
    }
    
    $result = ONNX\Math\multiply($tensor, 2);
    $data = $result->getData();
    if ($data[0] == 20 && $data[1] == 40 && $data[2] == 60) {
        echo "PASS: INT8 scalar multiplication works\n";
    } else {
        echo "FAIL: INT8 scalar multiplication: expected [20, 40, 60], got [" . implode(", ", $data) . "]\n";
    }
    
    $result = ONNX\Math\divide($tensor, 2);
    $data = $result->getData();
    if ($data[0] == 5 && $data[1] == 10 && $data[2] == 15) {
        echo "PASS: INT8 scalar division works\n";
    } else {
        echo "FAIL: INT8 scalar division: expected [5, 10, 15], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: INT8 scalar operations failed: " . $e->getMessage() . "\n";
}

// Test 2: INT16 scalar operations
try {
    $tensor = new ONNX\Tensor\Transient([3], [1000, 2000, 3000], ONNX\Tensor::INT16);
    
    $result = ONNX\Math\add($tensor, 500);
    echo "PASS: INT16 scalar addition works\n";
    
    $result = ONNX\Math\subtract($tensor, 500);
    echo "PASS: INT16 scalar subtraction works\n";
    
    $result = ONNX\Math\multiply($tensor, 2);
    echo "PASS: INT16 scalar multiplication works\n";
    
    $result = ONNX\Math\divide($tensor, 2);
    echo "PASS: INT16 scalar division works\n";
} catch (Error $e) {
    echo "FAIL: INT16 scalar operations failed: " . $e->getMessage() . "\n";
}

// Test 3: INT32 scalar operations
try {
    $tensor = new ONNX\Tensor\Transient([3], [100000, 200000, 300000], ONNX\Tensor::INT32);
    
    $result = ONNX\Math\add($tensor, 50000);
    echo "PASS: INT32 scalar addition works\n";
    
    $result = ONNX\Math\subtract($tensor, 50000);
    echo "PASS: INT32 scalar subtraction works\n";
    
    $result = ONNX\Math\multiply($tensor, 2);
    echo "PASS: INT32 scalar multiplication works\n";
    
    $result = ONNX\Math\divide($tensor, 2);
    echo "PASS: INT32 scalar division works\n";
} catch (Error $e) {
    echo "FAIL: INT32 scalar operations failed: " . $e->getMessage() . "\n";
}

// Test 4: UINT8 scalar operations
try {
    $tensor = new ONNX\Tensor\Transient([3], [100, 150, 200], ONNX\Tensor::UINT8);
    
    $result = ONNX\Math\add($tensor, 50);
    echo "PASS: UINT8 scalar addition works\n";
    
    $result = ONNX\Math\subtract($tensor, 50);
    echo "PASS: UINT8 scalar subtraction works\n";
    
    $result = ONNX\Math\multiply($tensor, 1);
    echo "PASS: UINT8 scalar multiplication works\n";
    
    $result = ONNX\Math\divide($tensor, 2);
    echo "PASS: UINT8 scalar division works\n";
} catch (Error $e) {
    echo "FAIL: UINT8 scalar operations failed: " . $e->getMessage() . "\n";
}

// Test 5: UINT16 scalar operations
try {
    $tensor = new ONNX\Tensor\Transient([3], [10000, 20000, 30000], ONNX\Tensor::UINT16);
    
    $result = ONNX\Math\add($tensor, 5000);
    echo "PASS: UINT16 scalar addition works\n";
    
    $result = ONNX\Math\subtract($tensor, 5000);
    echo "PASS: UINT16 scalar subtraction works\n";
    
    $result = ONNX\Math\multiply($tensor, 1);
    echo "PASS: UINT16 scalar multiplication works\n";
    
    $result = ONNX\Math\divide($tensor, 2);
    echo "PASS: UINT16 scalar division works\n";
} catch (Error $e) {
    echo "FAIL: UINT16 scalar operations failed: " . $e->getMessage() . "\n";
}

// Test 6: UINT32 scalar operations
try {
    $tensor = new ONNX\Tensor\Transient([3], [1000000, 2000000, 3000000], ONNX\Tensor::UINT32);
    
    $result = ONNX\Math\add($tensor, 500000);
    echo "PASS: UINT32 scalar addition works\n";
    
    $result = ONNX\Math\subtract($tensor, 500000);
    echo "PASS: UINT32 scalar subtraction works\n";
    
    $result = ONNX\Math\multiply($tensor, 1);
    echo "PASS: UINT32 scalar multiplication works\n";
    
    $result = ONNX\Math\divide($tensor, 2);
    echo "PASS: UINT32 scalar division works\n";
} catch (Error $e) {
    echo "FAIL: UINT32 scalar operations failed: " . $e->getMessage() . "\n";
}

// Test 7: Negation operations for various types
try {
    $tensor_int32 = new ONNX\Tensor\Transient([3], [10, -20, 30], ONNX\Tensor::INT32);
    $result = ONNX\Math\neg($tensor_int32);
    $data = $result->getData();
    if ($data[0] == -10 && $data[1] == 20 && $data[2] == -30) {
        echo "PASS: INT32 negation works\n";
    } else {
        echo "FAIL: INT32 negation: expected [-10, 20, -30], got [" . implode(", ", $data) . "]\n";
    }
    
    $tensor_uint8 = new ONNX\Tensor\Transient([3], [10, 20, 30], ONNX\Tensor::UINT8);
    $result = ONNX\Math\neg($tensor_uint8);
    echo "PASS: UINT8 negation works\n";
    
    $tensor_uint16 = new ONNX\Tensor\Transient([3], [1000, 2000, 3000], ONNX\Tensor::UINT16);
    $result = ONNX\Math\neg($tensor_uint16);
    echo "PASS: UINT16 negation works\n";
    
    $tensor_uint32 = new ONNX\Tensor\Transient([3], [100000, 200000, 300000], ONNX\Tensor::UINT32);
    $result = ONNX\Math\neg($tensor_uint32);
    echo "PASS: UINT32 negation works\n";
    
} catch (Error $e) {
    echo "FAIL: Negation operations failed: " . $e->getMessage() . "\n";
}

echo "Scalar operations with integer types testing completed!\n";
?>
--EXPECT--
=== Testing Scalar Operations with Integer Types ===
PASS: INT8 scalar addition works
PASS: INT8 scalar subtraction works
PASS: INT8 scalar multiplication works
PASS: INT8 scalar division works
PASS: INT16 scalar addition works
PASS: INT16 scalar subtraction works
PASS: INT16 scalar multiplication works
PASS: INT16 scalar division works
PASS: INT32 scalar addition works
PASS: INT32 scalar subtraction works
PASS: INT32 scalar multiplication works
PASS: INT32 scalar division works
PASS: UINT8 scalar addition works
PASS: UINT8 scalar subtraction works
PASS: UINT8 scalar multiplication works
PASS: UINT8 scalar division works
PASS: UINT16 scalar addition works
PASS: UINT16 scalar subtraction works
PASS: UINT16 scalar multiplication works
PASS: UINT16 scalar division works
PASS: UINT32 scalar addition works
PASS: UINT32 scalar subtraction works
PASS: UINT32 scalar multiplication works
PASS: UINT32 scalar division works
PASS: INT32 negation works
PASS: UINT8 negation works
PASS: UINT16 negation works
PASS: UINT32 negation works
Scalar operations with integer types testing completed!
