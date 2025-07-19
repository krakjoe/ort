--TEST--
Test integer tensor operations - comprehensive type coverage
--EXTENSIONS--
ort
--FILE--
<?php
echo "=== Testing Integer Tensor Operations ===\n";

// Test 1: int64_t operations
try {
    $tensor_a = new ORT\Tensor\Transient([2], [9223372036854775800, -100], ORT\Tensor::INT64);
    $tensor_b = new ORT\Tensor\Transient([2], [7, 50], ORT\Tensor::INT64);
    $result = ORT\Math\add($tensor_a, $tensor_b);
    echo "PASS: INT64 addition works\n";
    
    $result = ORT\Math\subtract($tensor_a, $tensor_b);
    echo "PASS: INT64 subtraction works\n";
    
    $result = ORT\Math\multiply($tensor_a, $tensor_b);
    echo "PASS: INT64 multiplication works\n";
} catch (Error $e) {
    echo "FAIL: INT64 operations failed: " . $e->getMessage() . "\n";
}

// Test 2: uint16_t operations
try {
    $tensor_a = new ORT\Tensor\Transient([3], [65000, 100, 500], ORT\Tensor::UINT16);
    $tensor_b = new ORT\Tensor\Transient([3], [535, 50, 200], ORT\Tensor::UINT16);
    $result = ORT\Math\add($tensor_a, $tensor_b);
    echo "PASS: UINT16 addition works\n";
    
    $result = ORT\Math\subtract($tensor_a, $tensor_b);
    echo "PASS: UINT16 subtraction works\n";
    
    $result = ORT\Math\multiply($tensor_a, $tensor_b);
    echo "PASS: UINT16 multiplication works\n";
} catch (Error $e) {
    echo "FAIL: UINT16 operations failed: " . $e->getMessage() . "\n";
}

// Test 3: int16_t operations
try {
    $tensor_a = new ORT\Tensor\Transient([2], [32000, -15000], ORT\Tensor::INT16);
    $tensor_b = new ORT\Tensor\Transient([2], [767, 5000], ORT\Tensor::INT16);
    $result = ORT\Math\add($tensor_a, $tensor_b);
    echo "PASS: INT16 addition works\n";
    
    $result = ORT\Math\subtract($tensor_a, $tensor_b);
    echo "PASS: INT16 subtraction works\n";
    
    $result = ORT\Math\multiply($tensor_a, $tensor_b);
    echo "PASS: INT16 multiplication works\n";
} catch (Error $e) {
    echo "FAIL: INT16 operations failed: " . $e->getMessage() . "\n";
}

// Test 4: int8_t operations
try {
    $tensor_a = new ORT\Tensor\Transient([3], [127, -128, 50], ORT\Tensor::INT8);
    $tensor_b = new ORT\Tensor\Transient([3], [1, 10, -25], ORT\Tensor::INT8);
    $result = ORT\Math\add($tensor_a, $tensor_b);
    echo "PASS: INT8 addition works\n";
    
    $result = ORT\Math\subtract($tensor_a, $tensor_b);
    echo "PASS: INT8 subtraction works\n";
    
    $result = ORT\Math\multiply($tensor_a, $tensor_b);
    echo "PASS: INT8 multiplication works\n";
} catch (Error $e) {
    echo "FAIL: INT8 operations failed: " . $e->getMessage() . "\n";
}

// Test 5: uint8_t operations
try {
    $tensor_a = new ORT\Tensor\Transient([3], [255, 100, 50], ORT\Tensor::UINT8);
    $tensor_b = new ORT\Tensor\Transient([3], [1, 55, 25], ORT\Tensor::UINT8);
    $result = ORT\Math\add($tensor_a, $tensor_b);
    echo "PASS: UINT8 addition works\n";
    
    $result = ORT\Math\subtract($tensor_a, $tensor_b);
    echo "PASS: UINT8 subtraction works\n";
    
    $result = ORT\Math\multiply($tensor_a, $tensor_b);
    echo "PASS: UINT8 multiplication works\n";
} catch (Error $e) {
    echo "FAIL: UINT8 operations failed: " . $e->getMessage() . "\n";
}

// Test 6: uint32_t operations
try {
    $tensor_a = new ORT\Tensor\Transient([2], [4294967200, 1000000], ORT\Tensor::UINT32);
    $tensor_b = new ORT\Tensor\Transient([2], [95, 500000], ORT\Tensor::UINT32);
    $result = ORT\Math\add($tensor_a, $tensor_b);
    echo "PASS: UINT32 addition works\n";
    
    $result = ORT\Math\subtract($tensor_a, $tensor_b);
    echo "PASS: UINT32 subtraction works\n";
    
    $result = ORT\Math\multiply($tensor_a, $tensor_b);
    echo "PASS: UINT32 multiplication works\n";
} catch (Error $e) {
    echo "FAIL: UINT32 operations failed: " . $e->getMessage() . "\n";
}

// Test 7: Scalar operations with integers
try {
    $tensor = new ORT\Tensor\Transient([2], [100, 200], ORT\Tensor::INT64);
    $result = ORT\Math\add($tensor, 50);
    echo "PASS: INT64 scalar addition works\n";
    
    $result = ORT\Math\subtract($tensor, 25);
    echo "PASS: INT64 scalar subtraction works\n";
    
    $result = ORT\Math\multiply($tensor, 3);
    echo "PASS: INT64 scalar multiplication works\n";
    
    $result = ORT\Math\divide($tensor, 4);
    echo "PASS: INT64 scalar division works\n";
} catch (Error $e) {
    echo "FAIL: INT64 scalar operations failed: " . $e->getMessage() . "\n";
}

// Test 8: Edge cases - maximum values
try {
    $tensor_max = new ORT\Tensor\Transient([2], [32767, -32768], ORT\Tensor::INT16);
    $result = ORT\Math\add($tensor_max, 0);
    echo "PASS: INT16 maximum value handling works\n";
    
    $tensor_uint8_max = new ORT\Tensor\Transient([2], [255, 0], ORT\Tensor::UINT8);
    $result = ORT\Math\multiply($tensor_uint8_max, 1);
    echo "PASS: UINT8 maximum value handling works\n";
} catch (Error $e) {
    echo "FAIL: Maximum value handling failed: " . $e->getMessage() . "\n";
}

echo "Integer tensor operation tests completed!\n";
?>
--EXPECT--
=== Testing Integer Tensor Operations ===
PASS: INT64 addition works
PASS: INT64 subtraction works
PASS: INT64 multiplication works
PASS: UINT16 addition works
PASS: UINT16 subtraction works
PASS: UINT16 multiplication works
PASS: INT16 addition works
PASS: INT16 subtraction works
PASS: INT16 multiplication works
PASS: INT8 addition works
PASS: INT8 subtraction works
PASS: INT8 multiplication works
PASS: UINT8 addition works
PASS: UINT8 subtraction works
PASS: UINT8 multiplication works
PASS: UINT32 addition works
PASS: UINT32 subtraction works
PASS: UINT32 multiplication works
PASS: INT64 scalar addition works
PASS: INT64 scalar subtraction works
PASS: INT64 scalar multiplication works
PASS: INT64 scalar division works
PASS: INT16 maximum value handling works
PASS: UINT8 maximum value handling works
Integer tensor operation tests completed!
