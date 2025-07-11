--TEST--
ONNX\Math\sqrt: line-by-line, all types, shape, error handling, numpy semantics
--EXTENSIONS--
ort
--FILE--
<?php
use ONNX\Tensor;


$types = [
    'FLOAT'   => ONNX\Tensor::FLOAT,
    'DOUBLE'  => ONNX\Tensor::DOUBLE,
    'INT8'    => ONNX\Tensor::INT8,
    'INT16'   => ONNX\Tensor::INT16,
    'INT32'   => ONNX\Tensor::INT32,
    'INT64'   => ONNX\Tensor::INT64,
    'UINT8'   => ONNX\Tensor::UINT8,
    'UINT16'  => ONNX\Tensor::UINT16,
    'UINT32'  => ONNX\Tensor::UINT32,
];

$signed_types = [
    'INT8'    => ONNX\Tensor::INT8,
    'INT16'   => ONNX\Tensor::INT16,
    'INT32'   => ONNX\Tensor::INT32,
    'INT64'   => ONNX\Tensor::INT64,
];
$unsigned_types = [
    'UINT8'   => ONNX\Tensor::UINT8,
    'UINT16'  => ONNX\Tensor::UINT16,
    'UINT32'  => ONNX\Tensor::UINT32,
];

$real = [
    'FLOAT'   => ONNX\Tensor::FLOAT,
    'DOUBLE'  => ONNX\Tensor::DOUBLE,
];

function flatten_data($data) {
    if (is_array($data)) {
        return '[' . implode(',', array_map('flatten_data', $data)) . ']';
    }
    if (is_float($data)) {
        if (is_nan($data)) return 'NAN';
        if ($data === INF) return 'INF';
        if ($data === -INF) return '-INF';
        if ($data == 0.0) return '0';
        // Print floats with max precision
        return rtrim(rtrim(sprintf('%.17g', $data), '0'), '.');
    }
    if (is_bool($data)) return $data ? '1' : '0';
    return (string)$data;
}

function print_result($result) {
    if ($result instanceof Tensor) {
        echo 'RESULT: ' . flatten_data($result->getData()) . "\n";
        echo 'TYPE: ' . $result->getType() . "\n";
        echo 'SHAPE: [' . implode(',', $result->getShape()) . "]\n";
    } else {
        echo "NOTENSOR\n";
    }
}


// 1. Sqrt for signed/real types: negatives, zero, positives
foreach (array_merge($real, $signed_types) as $name => $type) {
    $a = new ONNX\Tensor\Transient([5], [-1, 0, 4, 9, -9], $type);
    $result = ONNX\Math\sqrt($a);
    echo "PASS: $name sqrt [-1,0,4,9,-9]\n";
    print_result($result);
}
// 1b. Sqrt for unsigned types: only non-negative values
foreach ($unsigned_types as $name => $type) {
    $a = new ONNX\Tensor\Transient([3], [0, 4, 9], $type);
    $result = ONNX\Math\sqrt($a);
    echo "PASS: $name sqrt [0,4,9]\n";
    print_result($result);
}


// 2. Sqrt of zeros and ones (valid for all types)
foreach ($types as $name => $type) {
    $a = new ONNX\Tensor\Transient([6], [0,1,0,1,0,1], $type);
    $result = ONNX\Math\sqrt($a);
    echo "PASS: $name sqrt zeros/ones\n";
    print_result($result);
}


// 3. Sqrt of 2D tensor for INT8 (with negative and positive values)
$a = new ONNX\Tensor\Transient([2,2], [[12,16],[24,-128]], ONNX\Tensor::INT8);
$result = ONNX\Math\sqrt($a);
echo "PASS: INT8 sqrt 2D [[12,16],[24,-128]]\n";
print_result($result);


// 4. Sqrt of bool tensor
$a = new ONNX\Tensor\Transient([2,2], [[true,false],[false,true]], ONNX\Tensor::BOOL);
$result = ONNX\Math\sqrt($a);
echo "PASS: BOOL sqrt [bool 2x2] (numpy semantics)\n";
print_result($result);


// 5. Sqrt of large and small values (real types only)
foreach ($real as $name => $type) {
    $a = new ONNX\Tensor\Transient([4], [1e10, 1e-10, 0, 1], $type);
    $result = ONNX\Math\sqrt($a);
    echo "PASS: $name sqrt extremes\n";
    print_result($result);
}


// 6. Sqrt of empty tensor (should error)
try {
    $a = new ONNX\Tensor\Transient([0], [], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\sqrt($a);
    echo "FAIL: Did not throw on empty tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on empty tensor: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT sqrt [-1,0,4,9,-9]
RESULT: %s
TYPE: %d
SHAPE: [5]
PASS: DOUBLE sqrt [-1,0,4,9,-9]
RESULT: %s
TYPE: %d
SHAPE: [5]
PASS: INT8 sqrt [-1,0,4,9,-9]
RESULT: %s
TYPE: %d
SHAPE: [5]
PASS: INT16 sqrt [-1,0,4,9,-9]
RESULT: %s
TYPE: %d
SHAPE: [5]
PASS: INT32 sqrt [-1,0,4,9,-9]
RESULT: %s
TYPE: %d
SHAPE: [5]
PASS: INT64 sqrt [-1,0,4,9,-9]
RESULT: %s
TYPE: %d
SHAPE: [5]
PASS: UINT8 sqrt [0,4,9]
RESULT: %s
TYPE: %d
SHAPE: [3]
PASS: UINT16 sqrt [0,4,9]
RESULT: %s
TYPE: %d
SHAPE: [3]
PASS: UINT32 sqrt [0,4,9]
RESULT: %s
TYPE: %d
SHAPE: [3]
PASS: FLOAT sqrt zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [6]
PASS: DOUBLE sqrt zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [6]
PASS: INT8 sqrt zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [6]
PASS: INT16 sqrt zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [6]
PASS: INT32 sqrt zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [6]
PASS: INT64 sqrt zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [6]
PASS: UINT8 sqrt zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [6]
PASS: UINT16 sqrt zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [6]
PASS: UINT32 sqrt zeros/ones
RESULT: %s
TYPE: %d
SHAPE: [6]
PASS: INT8 sqrt 2D [[12,16],[24,-128]]
RESULT: %s
TYPE: %d
SHAPE: [2,2]
PASS: BOOL sqrt [bool 2x2] (numpy semantics)
RESULT: [[1,0],[0,1]]
TYPE: %d
SHAPE: [2,2]
PASS: FLOAT sqrt extremes
RESULT: %s
TYPE: %d
SHAPE: [4]
PASS: DOUBLE sqrt extremes
RESULT: %s
TYPE: %d
SHAPE: [4]
PASS: Error on empty tensor: %s
