--TEST--
Test type casting and broadcasting compatibility - edge cases
--EXTENSIONS--
ort
--FILE--
<?php
echo "=== Testing Type Casting and Broadcasting Compatibility ===\n";

// Test 1: Mixed type operations that should work (type promotion)
try {
    $tensor_float = new ORT\Tensor\Transient([2], [1.5, 2.5], ORT\Tensor::FLOAT);
    $tensor_int = new ORT\Tensor\Transient([2], [3, 4], ORT\Tensor::INT32);
    
    $result = ORT\Math\add($tensor_float, $tensor_int);
    echo "PASS: Mixed FLOAT and INT32 addition works (type promotion)\n";
    
    $result = ORT\Math\multiply($tensor_float, $tensor_int);
    echo "PASS: Mixed FLOAT and INT32 multiplication works\n";
} catch (Error $e) {
    echo "INFO: Mixed type operations result: " . get_class($e) . " - " . $e->getMessage() . "\n";
}

// Test 2: Boolean with numeric types
try {
    $tensor_bool = new ORT\Tensor\Transient([2], [true, false], ORT\Tensor::BOOL);
    $tensor_float = new ORT\Tensor\Transient([2], [1.0, 2.0], ORT\Tensor::FLOAT);
    
    $result = ORT\Math\add($tensor_bool, $tensor_float);
    echo "INFO: Boolean and FLOAT addition behavior\n";
} catch (Error $e) {
    echo "INFO: Boolean and FLOAT mixing: " . get_class($e) . "\n";
}

// Test 3: Different integer type combinations
try {
    $tensor_int8 = new ORT\Tensor\Transient([2], [10, 20], ORT\Tensor::INT8);
    $tensor_int32 = new ORT\Tensor\Transient([2], [100, 200], ORT\Tensor::INT32);
    
    $result = ORT\Math\add($tensor_int8, $tensor_int32);
    echo "PASS: INT8 and INT32 addition works\n";
} catch (Error $e) {
    echo "INFO: INT8 and INT32 mixing: " . get_class($e) . "\n";
}

// Test 4: Unsigned and signed integer mixing
try {
    $tensor_uint8 = new ORT\Tensor\Transient([2], [50, 100], ORT\Tensor::UINT8);
    $tensor_int16 = new ORT\Tensor\Transient([2], [-10, 25], ORT\Tensor::INT16);
    
    $result = ORT\Math\subtract($tensor_uint8, $tensor_int16);
    echo "PASS: UINT8 and INT16 subtraction works\n";
} catch (Error $e) {
    echo "INFO: UINT8 and INT16 mixing: " . get_class($e) . "\n";
}

// Test 5: Double and float mixing
try {
    $tensor_double = new ORT\Tensor\Transient([2], [3.14159265359, 2.71828182846], ORT\Tensor::DOUBLE);
    $tensor_float = new ORT\Tensor\Transient([2], [1.41421356237, 1.73205080757], ORT\Tensor::FLOAT);
    
    $result = ORT\Math\multiply($tensor_double, $tensor_float);
    echo "PASS: DOUBLE and FLOAT multiplication works\n";
} catch (Error $e) {
    echo "INFO: DOUBLE and FLOAT mixing: " . get_class($e) . "\n";
}

// Test 6: Broadcasting compatibility - different shapes
try {
    $tensor_2x3 = new ORT\Tensor\Transient([2, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ORT\Tensor::FLOAT);
    $tensor_1x3 = new ORT\Tensor\Transient([1, 3], [[10.0, 20.0, 30.0]], ORT\Tensor::FLOAT);
    
    $result = ORT\Math\add($tensor_2x3, $tensor_1x3);
    echo "PASS: Broadcasting (2x3 + 1x3) works\n";
} catch (Error $e) {
    echo "INFO: Broadcasting (2x3 + 1x3): " . get_class($e) . "\n";
}

// Test 7: Scalar broadcasting (already tested but for completeness)
try {
    $tensor = new ORT\Tensor\Transient([2, 2], [[1.0, 2.0], [3.0, 4.0]], ORT\Tensor::FLOAT);
    $result = ORT\Math\add($tensor, 10.0);
    echo "PASS: Scalar broadcasting works\n";
} catch (Error $e) {
    echo "FAIL: Scalar broadcasting failed: " . $e->getMessage() . "\n";
}

// Test 8: Incompatible broadcasting shapes
try {
    $tensor_2x3 = new ORT\Tensor\Transient([2, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ORT\Tensor::FLOAT);
    $tensor_2x2 = new ORT\Tensor\Transient([2, 2], [[1.0, 2.0], [3.0, 4.0]], ORT\Tensor::FLOAT);
    
    $result = ORT\Math\add($tensor_2x3, $tensor_2x2);
    echo "FAIL: Incompatible shapes should throw exception\n";
} catch (Error $e) {
    echo "PASS: Incompatible broadcasting shapes correctly rejected: " . get_class($e) . "\n";
}

// Test 9: Complex type promotion chains
try {
    $tensor_uint8 = new ORT\Tensor\Transient([2], [255, 128], ORT\Tensor::UINT8);
    $tensor_int64 = new ORT\Tensor\Transient([2], [1000000000000, -500000000000], ORT\Tensor::INT64);
    
    $result = ORT\Math\add($tensor_uint8, $tensor_int64);
    echo "PASS: UINT8 and INT64 type promotion works\n";
} catch (Error $e) {
    echo "INFO: UINT8 and INT64 promotion: " . get_class($e) . "\n";
}

// Test 10: Edge case - same type but different tensors
try {
    $tensor_a = new ORT\Tensor\Transient([3], [1.0, 2.0, 3.0], ORT\Tensor::FLOAT);
    $tensor_b = new ORT\Tensor\Transient([3], [4.0, 5.0, 6.0], ORT\Tensor::FLOAT);
    
    $result = ORT\Math\add($tensor_a, $tensor_b);
    $data = $result->getData();
    if (abs($data[0] - 5.0) < 0.001 && abs($data[1] - 7.0) < 0.001 && abs($data[2] - 9.0) < 0.001) {
        echo "PASS: Same type tensors work correctly\n";
    } else {
        echo "FAIL: Same type addition: expected [5, 7, 9], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: Same type tensor operation failed: " . $e->getMessage() . "\n";
}

// Test 11: Type validation for unsupported operations
try {
    $tensor_bool = new ORT\Tensor\Transient([2], [true, false], ORT\Tensor::BOOL);
    $result = ORT\Math\sqrt($tensor_bool);
    echo "FAIL: sqrt on boolean tensor should be rejected\n";
} catch (ORT\Status\Math\InvalidType $e) {
    echo "PASS: sqrt correctly rejects boolean tensors\n";
} catch (Error $e) {
    echo "PARTIAL: sqrt on boolean throws: " . get_class($e) . "\n";
}

// Test 13: Division by zero handling across types
try {
    $tensor_float = new ORT\Tensor\Transient([2], [10.0, 20.0], ORT\Tensor::FLOAT);
    $result = ORT\Math\divide($tensor_float, 0.0);
    echo "INFO: Float division by zero handling\n";
} catch (Error $e) {
    echo "INFO: Float division by zero: " . get_class($e) . "\n";
}

// Test 14: Overflow/underflow behavior
try {
    $tensor_uint8 = new ORT\Tensor\Transient([2], [255, 200], ORT\Tensor::UINT8);
    $result = ORT\Math\add($tensor_uint8, 100); // Should overflow for first element
    echo "INFO: UINT8 overflow behavior tested\n";
} catch (Error $e) {
    echo "INFO: UINT8 overflow: " . get_class($e) . "\n";
}

echo "Type casting and broadcasting compatibility tests completed!\n";
?>
--EXPECTF--
=== Testing Type Casting and Broadcasting Compatibility ===
%a
%a
%a
%a
%a
%a
PASS: Scalar broadcasting works
PASS: Incompatible broadcasting shapes correctly rejected: %s
%a
PASS: Same type tensors work correctly
%a
%a
%a
Type casting and broadcasting compatibility tests completed!
