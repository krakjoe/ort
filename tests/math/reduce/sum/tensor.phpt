--TEST--
ONNX\Math\reduce\tensor\sum: all types, shape, and error handling
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
    /* compliance, sum(unsigned) -> uint64
        we don't support uint64, so cannot
        support these operations with numpy semantics
    'UINT8'   => ONNX\Tensor::UINT8,
    'UINT16'  => ONNX\Tensor::UINT16,
    'UINT32'  => ONNX\Tensor::UINT32,
    */
    'BOOL'    => ONNX\Tensor::BOOL,
];

function print_result($result) {
    if ($result instanceof Tensor) {
        echo 'RESULT: ' . json_encode($result->getData()) . "\n";
        echo 'TYPE: ' . $result->getType() . "\n";
        echo 'SHAPE: [' . implode(',', $result->getShape()) . "]\n";
    } else {
        echo "NOTENSOR\n";
    }
}

// 1. Basic sum for all types (2x3)
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\reduce\tensor\sum($a);
        echo "PASS: $name sum\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name sum: ".$e->getMessage()."\n";
    }
}

// 2. Error: empty tensor
try {
    $a = new ONNX\Tensor\Transient([0], [], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\reduce\tensor\sum($a);
    echo "FAIL: Did not throw on empty tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on empty tensor: ".$e->getMessage()."\n";
}

// 3. Error: bool sum with non-binary values
try {
    $a = new ONNX\Tensor\Transient([2,2], [[1,2],[3,4]], ONNX\Tensor::BOOL);
    $result = ONNX\Math\reduce\tensor\sum($a);
    echo "FAIL: Did not throw on non-binary bool tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on non-binary bool tensor: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT sum
RESULT: [21]
TYPE: %d
SHAPE: []
PASS: DOUBLE sum
RESULT: [21]
TYPE: %d
SHAPE: []
PASS: INT8 sum
RESULT: [21]
TYPE: %d
SHAPE: []
PASS: INT16 sum
RESULT: [21]
TYPE: %d
SHAPE: []
PASS: INT32 sum
RESULT: [21]
TYPE: %d
SHAPE: []
PASS: INT64 sum
RESULT: [21]
TYPE: %d
SHAPE: []
PASS: BOOL sum
RESULT: [3]
TYPE: %d
SHAPE: []
PASS: Error on empty tensor: shape information must be an array of positive integers
PASS: Error on non-binary bool tensor: validation of data according to the shape provided has failed
