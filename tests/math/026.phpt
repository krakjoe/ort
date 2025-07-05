--TEST--
Test sum operation with axis parameter - comprehensive coverage
--EXTENSIONS--
ort
--FILE--
<?php
echo "=== Testing Sum Operation with Axis Parameter ===\n";

// Test 1: Sum all elements (no axis specified)
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, null);
    $data = $result->getData();
    if (abs($data[0] - 21.0) < 0.001) {
        echo "PASS: Sum all elements works (got " . $data[0] . ")\n";
    } else {
        echo "FAIL: Sum all elements: expected 21.0, got " . $data[0] . "\n";
    }
} catch (Error $e) {
    echo "FAIL: Sum all elements failed: " . $e->getMessage() . "\n";
}

// Test 2: Sum along axis 0 (rows)
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, 0);
    $data = $result->getData();
    $shape = $result->getShape();
    // Sum along axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
    if (count($data) == 3 && abs($data[0] - 5.0) < 0.001 && abs($data[1] - 7.0) < 0.001 && abs($data[2] - 9.0) < 0.001) {
        echo "PASS: Sum along axis 0 works: [" . implode(", ", $data) . "]\n";
    } else {
        echo "FAIL: Sum along axis 0: expected [5, 7, 9], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: Sum along axis 0 failed: " . $e->getMessage() . "\n";
}

// Test 3: Sum along axis 1 (columns)
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, 1);
    $data = $result->getData();
    // Sum along axis 1: [1+2+3, 4+5+6] = [6, 15]
    if (count($data) == 2 && abs($data[0] - 6.0) < 0.001 && abs($data[1] - 15.0) < 0.001) {
        echo "PASS: Sum along axis 1 works: [" . implode(", ", $data) . "]\n";
    } else {
        echo "FAIL: Sum along axis 1: expected [6, 15], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: Sum along axis 1 failed: " . $e->getMessage() . "\n";
}

// Test 4: Negative axis indexing
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, -1);
    $data = $result->getData();
    // axis=-1 is same as axis=1 for 2D tensor
    if (count($data) == 2 && abs($data[0] - 6.0) < 0.001 && abs($data[1] - 15.0) < 0.001) {
        echo "PASS: Sum with negative axis indexing works\n";
    } else {
        echo "FAIL: Sum with negative axis: expected [6, 15], got [" . implode(", ", $data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: Sum with negative axis failed: " . $e->getMessage() . "\n";
}

// Test 5: keepdims parameter
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, 0, true);
    $shape = $result->getShape();
    // keepdims=true should preserve dimensions as [1, 3]
    if (count($shape) == 2 && $shape[0] == 1 && $shape[1] == 3) {
        echo "PASS: Sum with keepdims=true works (shape: [" . implode(", ", $shape) . "])\n";
    } else {
        echo "FAIL: Sum with keepdims: expected shape [1, 3], got [" . implode(", ", $shape) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: Sum with keepdims failed: " . $e->getMessage() . "\n";
}

// Test 6: Sum with double precision
try {
    $tensor = new ONNX\Tensor\Transient([2, 2], [[1.5, 2.7], [3.1, 4.9]], ONNX\Tensor::DOUBLE);
    $result = ONNX\Math\sum($tensor, null);
    $data = $result->getData();
    if (abs($data[0] - 12.2) < 0.001) {
        echo "PASS: Sum with double works (got " . $data[0] . ")\n";
    } else {
        echo "FAIL: Sum with double: expected 12.2, got " . $data[0] . "\n";
    }
} catch (Error $e) {
    echo "FAIL: Sum with double failed: " . $e->getMessage() . "\n";
}

// Test 7: Sum with INT32
try {
    $tensor = new ONNX\Tensor\Transient([2, 2], [[10, 20], [30, 40]], ONNX\Tensor::INT32);
    $result = ONNX\Math\sum($tensor, null);
    $data = $result->getData();
    if ($data[0] == 100) {
        echo "PASS: Sum with INT32 works (got " . $data[0] . ")\n";
    } else {
        echo "FAIL: Sum with INT32: expected 100, got " . $data[0] . "\n";
    }
} catch (Error $e) {
    echo "FAIL: Sum with INT32 failed: " . $e->getMessage() . "\n";
}

// Test 8: Sum with boolean (using true/false values)
try {
    $tensor = new ONNX\Tensor\Transient([2], [true, false], ONNX\Tensor::BOOL);
    $result = ONNX\Math\sum($tensor, null);
    $data = $result->getData();
    if ($data[0] == 1) {
        echo "PASS: Sum with boolean works (got " . $data[0] . ")\n";
    } else {
        echo "FAIL: Sum with boolean: expected 1, got " . $data[0] . "\n";
    }
} catch (Error $e) {
    echo "FAIL: Sum with boolean failed: " . $e->getMessage() . "\n";
}

// Test 9: Invalid axis error handling
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, 5); // axis 5 is invalid for 2D tensor
    echo "FAIL: Invalid axis should throw exception\n";
} catch (ONNX\Status\Tensor\InvalidShape $e) {
    echo "PASS: Invalid axis correctly throws InvalidShape exception\n";
} catch (Error $e) {
    echo "FAIL: Invalid axis threw wrong exception: " . $e->getMessage() . "\n";
}

// Test 10: Invalid axis type error handling
try {
    $tensor = new ONNX\Tensor\Transient([2, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, "invalid"); // axis must be integer
    echo "FAIL: Invalid axis type should throw exception\n";
} catch (ONNX\Status\Tensor\InvalidShape $e) {
    echo "PASS: Invalid axis type correctly throws InvalidShape exception\n";
} catch (Error $e) {
    echo "FAIL: Invalid axis type threw wrong exception: " . $e->getMessage() . "\n";
}

// Test 11: 3D tensor sum with axis
try {
    $tensor = new ONNX\Tensor\Transient([2, 2, 2], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sum($tensor, 0);
    $data = $result->getData();
    // Sum along axis 0: [[1+5, 2+6], [3+7, 4+8]] = [[6, 8], [10, 12]]
    // getData() might return nested arrays, so we need to flatten or check the structure
    $flattened = [];
    if (is_array($data[0])) {
        // If data is nested, flatten it
        foreach ($data as $row) {
            if (is_array($row)) {
                foreach ($row as $val) {
                    $flattened[] = $val;
                }
            } else {
                $flattened[] = $row;
            }
        }
    } else {
        $flattened = $data;
    }
    
    $expected = [6, 8, 10, 12];
    $match = true;
    if (count($flattened) != count($expected)) {
        $match = false;
    } else {
        for ($i = 0; $i < count($flattened); $i++) {
            if (abs($flattened[$i] - $expected[$i]) > 0.001) {
                $match = false;
                break;
            }
        }
    }
    
    if ($match) {
        echo "PASS: 3D tensor sum works: [" . implode(", ", $flattened) . "]\n";
    } else {
        echo "FAIL: 3D tensor sum: expected [" . implode(", ", $expected) . "], got [" . json_encode($data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: 3D tensor sum failed: " . $e->getMessage() . "\n";
}

echo "Sum operation with axis tests completed!\n";
?>
--EXPECT--
=== Testing Sum Operation with Axis Parameter ===
PASS: Sum all elements works (got 21)
PASS: Sum along axis 0 works: [5, 7, 9]
PASS: Sum along axis 1 works: [6, 15]
PASS: Sum with negative axis indexing works
PASS: Sum with keepdims=true works (shape: [1, 3])
PASS: Sum with double works (got 12.2)
PASS: Sum with INT32 works (got 100)
PASS: Sum with boolean works (got 1)
PASS: Invalid axis correctly throws InvalidShape exception
PASS: Invalid axis type correctly throws InvalidShape exception
PASS: 3D tensor sum works: [6, 8, 10, 12]
Sum operation with axis tests completed!
