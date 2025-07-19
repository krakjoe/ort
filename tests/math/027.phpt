--TEST--
Test matrix operations and validation - advanced coverage
--EXTENSIONS--
ort
--FILE--
<?php
echo "=== Testing Matrix Operations and Validation ===\n";

// Test 1: Basic 2x2 matrix multiplication
try {
    $matrix_a = new ORT\Tensor\Transient([2, 2], [[1.0, 2.0], [3.0, 4.0]], ORT\Tensor::FLOAT);
    $matrix_b = new ORT\Tensor\Transient([2, 2], [[5.0, 6.0], [7.0, 8.0]], ORT\Tensor::FLOAT);
    $result = ORT\Math\matmul($matrix_a, $matrix_b);
    $data = $result->getData();
    $shape = $result->getShape();
    
    // Helper function to flatten data if nested
    function flatten_matrix_data($data) {
        $flattened = [];
        if (is_array($data)) {
            foreach ($data as $element) {
                if (is_array($element)) {
                    $flattened = array_merge($flattened, flatten_matrix_data($element));
                } else {
                    $flattened[] = $element;
                }
            }
        } else {
            $flattened[] = $data;
        }
        return $flattened;
    }
    
    $flat_data = flatten_matrix_data($data);
    
    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    if (count($shape) == 2 && $shape[0] == 2 && $shape[1] == 2 &&
        count($flat_data) >= 4 &&
        abs($flat_data[0] - 19.0) < 0.001 && abs($flat_data[1] - 22.0) < 0.001 &&
        abs($flat_data[2] - 43.0) < 0.001 && abs($flat_data[3] - 50.0) < 0.001) {
        echo "PASS: 2x2 matrix multiplication works\n";
    } else {
        echo "FAIL: 2x2 matmul: expected [[19, 22], [43, 50]], got [" . implode(", ", $flat_data) . "] with shape [" . implode(", ", $shape) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: 2x2 matrix multiplication failed: " . $e->getMessage() . "\n";
}

// Test 2: Rectangular matrix multiplication (3x2 * 2x4)
try {
    $matrix_a = new ORT\Tensor\Transient([3, 2], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], ORT\Tensor::FLOAT);
    $matrix_b = new ORT\Tensor\Transient([2, 4], [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], ORT\Tensor::FLOAT);
    $result = ORT\Math\matmul($matrix_a, $matrix_b);
    $shape = $result->getShape();
    
    if (count($shape) == 2 && $shape[0] == 3 && $shape[1] == 4) {
        echo "PASS: Rectangular matrix multiplication (3x2 * 2x4) works, result shape: [" . implode(", ", $shape) . "]\n";
    } else {
        echo "FAIL: Rectangular matmul shape: expected [3, 4], got [" . implode(", ", $shape) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: Rectangular matrix multiplication failed: " . $e->getMessage() . "\n";
}

// Test 3: Double precision matrix multiplication
try {
    $matrix_a = new ORT\Tensor\Transient([2, 2], [[1.5, 2.5], [3.5, 4.5]], ORT\Tensor::DOUBLE);
    $matrix_b = new ORT\Tensor\Transient([2, 2], [[0.5, 1.5], [2.5, 3.5]], ORT\Tensor::DOUBLE);
    $result = ORT\Math\matmul($matrix_a, $matrix_b);
    echo "PASS: Double precision matrix multiplication works\n";
} catch (Error $e) {
    echo "FAIL: Double precision matrix multiplication failed: " . $e->getMessage() . "\n";
}

// Test 4: Matrix multiplication dimension mismatch error
try {
    $matrix_a = new ORT\Tensor\Transient([2, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], ORT\Tensor::FLOAT);
    $matrix_b = new ORT\Tensor\Transient([2, 2], [[1.0, 2.0], [3.0, 4.0]], ORT\Tensor::FLOAT);
    $result = ORT\Math\matmul($matrix_a, $matrix_b);
    echo "FAIL: Dimension mismatch should throw exception\n";
} catch (ORT\Status\Math\Error $e) {
    echo "PASS: Matrix dimension mismatch correctly throws Math Error\n";
} catch (Error $e) {
    echo "PARTIAL: Dimension mismatch throws exception: " . get_class($e) . "\n";
}

// Test 5: 3D tensor matmul rejection
try {
    $tensor_3d = new ORT\Tensor\Transient([2, 2, 2], [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], ORT\Tensor::FLOAT);
    $matrix = new ORT\Tensor\Transient([2, 2], [[1.0, 2.0], [3.0, 4.0]], ORT\Tensor::FLOAT);
    $result = ORT\Math\matmul($tensor_3d, $matrix);
    echo "FAIL: 3D tensor should be rejected for matmul\n";
} catch (ORT\Status\Math\Error $e) {
    echo "PASS: 3D tensor correctly rejected for matmul\n";
} catch (Error $e) {
    echo "PARTIAL: 3D tensor rejection throws: " . get_class($e) . "\n";
}

// Test 6: 1D vector matmul rejection
try {
    $vector = new ORT\Tensor\Transient([4], [1.0, 2.0, 3.0, 4.0], ORT\Tensor::FLOAT);
    $matrix = new ORT\Tensor\Transient([2, 2], [[1.0, 2.0], [3.0, 4.0]], ORT\Tensor::FLOAT);
    $result = ORT\Math\matmul($vector, $matrix);
    echo "FAIL: 1D vector should be rejected for matmul\n";
} catch (ORT\Status\Math\Error $e) {
    echo "PASS: 1D vector correctly rejected for matmul\n";
} catch (Error $e) {
    echo "PARTIAL: 1D vector rejection throws: " . get_class($e) . "\n";
}

// Test 7: Empty matrix handling
try {
    $matrix_a = new ORT\Tensor\Transient([0, 0], [], ORT\Tensor::FLOAT);
    $matrix_b = new ORT\Tensor\Transient([0, 0], [], ORT\Tensor::FLOAT);
    $result = ORT\Math\matmul($matrix_a, $matrix_b);
    echo "INFO: Empty matrix multiplication handling\n";
} catch (Error $e) {
    echo "INFO: Empty matrix throws exception: " . get_class($e) . "\n";
}

// Test 8: Large matrix multiplication (performance test)
try {
    // Create 10x5 matrix with proper nested structure
    $data_a = [];
    for ($i = 0; $i < 10; $i++) {
        $data_a[] = array_fill(0, 5, 1.0);
    }
    
    // Create 5x8 matrix with proper nested structure  
    $data_b = [];
    for ($i = 0; $i < 5; $i++) {
        $data_b[] = array_fill(0, 8, 0.5);
    }
    
    $matrix_a = new ORT\Tensor\Transient([10, 5], $data_a, ORT\Tensor::FLOAT);
    $matrix_b = new ORT\Tensor\Transient([5, 8], $data_b, ORT\Tensor::FLOAT);
    $result = ORT\Math\matmul($matrix_a, $matrix_b);
    $shape = $result->getShape();
    
    if (count($shape) == 2 && $shape[0] == 10 && $shape[1] == 8) {
        echo "PASS: Large matrix multiplication (10x5 * 5x8) works\n";
    } else {
        echo "FAIL: Large matrix shape: expected [10, 8], got [" . implode(", ", $shape) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: Large matrix multiplication failed: " . $e->getMessage() . "\n";
}

// Test 9: Identity matrix multiplication
try {
    $matrix = new ORT\Tensor\Transient([3, 3], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], ORT\Tensor::FLOAT);
    $identity = new ORT\Tensor\Transient([3, 3], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], ORT\Tensor::FLOAT);
    $result = ORT\Math\matmul($matrix, $identity);
    $data = $result->getData();
    $original_data = $matrix->getData();
    
    // Helper function to flatten data if nested
    function flatten_data($data) {
        $flattened = [];
        if (is_array($data)) {
            foreach ($data as $element) {
                if (is_array($element)) {
                    $flattened = array_merge($flattened, flatten_data($element));
                } else {
                    $flattened[] = $element;
                }
            }
        } else {
            $flattened[] = $data;
        }
        return $flattened;
    }
    
    $flat_data = flatten_data($data);
    $flat_original = flatten_data($original_data);
    
    $is_same = true;
    if (count($flat_data) == count($flat_original)) {
        for ($i = 0; $i < count($flat_data); $i++) {
            if (abs($flat_data[$i] - $flat_original[$i]) > 0.001) {
                $is_same = false;
                break;
            }
        }
    } else {
        $is_same = false;
    }
    
    if ($is_same) {
        echo "PASS: Identity matrix multiplication preserves original matrix\n";
    } else {
        echo "FAIL: Identity matrix multiplication changed values\n";
    }
} catch (Error $e) {
    echo "FAIL: Identity matrix multiplication failed: " . $e->getMessage() . "\n";
}

// Test 10: Zero matrix multiplication
try {
    $matrix = new ORT\Tensor\Transient([2, 2], [[1.0, 2.0], [3.0, 4.0]], ORT\Tensor::FLOAT);
    $zero_matrix = new ORT\Tensor\Transient([2, 2], [[0.0, 0.0], [0.0, 0.0]], ORT\Tensor::FLOAT);
    $result = ORT\Math\matmul($matrix, $zero_matrix);
    $data = $result->getData();
    
    // Helper function to flatten data if nested
    function flatten_zero_data($data) {
        $flattened = [];
        if (is_array($data)) {
            foreach ($data as $element) {
                if (is_array($element)) {
                    $flattened = array_merge($flattened, flatten_zero_data($element));
                } else {
                    $flattened[] = $element;
                }
            }
        } else {
            $flattened[] = $data;
        }
        return $flattened;
    }
    
    $flat_data = flatten_zero_data($data);
    
    $all_zero = true;
    foreach ($flat_data as $value) {
        if (abs($value) > 0.001) {
            $all_zero = false;
            break;
        }
    }
    
    if ($all_zero) {
        echo "PASS: Zero matrix multiplication results in zero matrix\n";
    } else {
        echo "FAIL: Zero matrix multiplication has non-zero values: [" . implode(", ", $flat_data) . "]\n";
    }
} catch (Error $e) {
    echo "FAIL: Zero matrix multiplication failed: " . $e->getMessage() . "\n";
}

echo "Matrix operations and validation tests completed!\n";
?>
--EXPECT--
=== Testing Matrix Operations and Validation ===
PASS: 2x2 matrix multiplication works
PASS: Rectangular matrix multiplication (3x2 * 2x4) works, result shape: [3, 4]
PASS: Double precision matrix multiplication works
PASS: Matrix dimension mismatch correctly throws Math Error
PASS: 3D tensor correctly rejected for matmul
PASS: 1D vector correctly rejected for matmul
INFO: Empty matrix throws exception: ORT\Status\Tensor\InvalidShape
PASS: Large matrix multiplication (10x5 * 5x8) works
PASS: Identity matrix multiplication preserves original matrix
PASS: Zero matrix multiplication results in zero matrix
Matrix operations and validation tests completed!
