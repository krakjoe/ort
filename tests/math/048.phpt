--TEST--
ONNX\Math\recip function coverage
--EXTENSIONS--
ort
--FILE--
<?php
// Test recip of positive, negative, and zero values (float)
$tensor = new ONNX\Tensor\Transient([4], [2.0, -4.0, 0.5, 0.0], ONNX\Tensor::FLOAT);
$result = ONNX\Math\recip($tensor);
echo "Reciprocal (float): ";
$data = $result->getData();
foreach ($data as $v) {
    if (is_infinite($v)) {
        echo ($v > 0 ? 'INF' : '-INF') . ' ';
    } elseif (is_nan($v)) {
        echo 'NaN ';
    } else {
        echo round($v, 3) . ' ';
    }
}
echo "\n";

// Test recip of positive, negative, and zero values (double)
$tensor = new ONNX\Tensor\Transient([4], [2.0, -4.0, 0.5, 0.0], ONNX\Tensor::DOUBLE);
$result = ONNX\Math\recip($tensor);
echo "Reciprocal (double): ";
$data = $result->getData();
foreach ($data as $v) {
    if (is_infinite($v)) {
        echo ($v > 0 ? 'INF' : '-INF') . ' ';
    } elseif (is_nan($v)) {
        echo 'NaN ';
    } else {
        echo round($v, 3) . ' ';
    }
}
echo "\n";

// Test recip of integer values (should throw)
try {
    $tensor_int = new ONNX\Tensor\Transient([3], [1, -2, 0], ONNX\Tensor::INT32);
    $result_int = ONNX\Math\recip($tensor_int);
    echo "FAIL: No exception for int recip\n";
} catch (Throwable $e) {
    echo "PASS: Exception for int recip: " . get_class($e) . "\n";
}

// Test error handling: invalid input (not a tensor)
try {
    ONNX\Math\recip(42);
    echo "FAIL: No exception for invalid input\n";
} catch (Throwable $e) {
    echo "PASS: Exception for invalid input: " . get_class($e) . "\n";
}
?>
--EXPECTF--
Reciprocal (float): 0.5 -0.25 2 INF 
Reciprocal (double): 0.5 -0.25 2 INF 
PASS: Exception for int recip: ONNX\Status\Math\InvalidType
PASS: Exception for invalid input: %s
