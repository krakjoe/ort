--TEST--
ONNX\Math\reduce\tensor\mean: all types, shape, and error handling
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

// 1. Basic mean for all types (2x3)
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\reduce\tensor\mean($a);
        echo "PASS: $name mean\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name mean: ".$e->getMessage()."\n";
    }
}

// 2. Error: empty tensor
try {
    $a = new ONNX\Tensor\Transient([0], [], ONNX\Tensor::FLOAT);
    $result = ONNX\Math\reduce\tensor\mean($a);
    echo "FAIL: Did not throw on empty tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on empty tensor: ".$e->getMessage()."\n";
}

// 3. Error: bool mean with non-binary values
try {
    $a = new ONNX\Tensor\Transient([2,2], [[1,2],[3,4]], ONNX\Tensor::BOOL);
    $result = ONNX\Math\reduce\tensor\mean($a);
    echo "FAIL: Did not throw on non-binary bool tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on non-binary bool tensor: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT mean
RESULT: [3.5]
TYPE: %d
SHAPE: []
PASS: DOUBLE mean
RESULT: [3.5]
TYPE: %d
SHAPE: []
PASS: INT8 mean
RESULT: [3.5]
TYPE: %d
SHAPE: []
PASS: INT16 mean
RESULT: [3.5]
TYPE: %d
SHAPE: []
PASS: INT32 mean
RESULT: [3.5]
TYPE: %d
SHAPE: []
PASS: INT64 mean
RESULT: [3.5]
TYPE: %d
SHAPE: []
PASS: UINT8 mean
RESULT: [3.5]
TYPE: %d
SHAPE: []
PASS: UINT16 mean
RESULT: [3.5]
TYPE: %d
SHAPE: []
PASS: UINT32 mean
RESULT: [3.5]
TYPE: %d
SHAPE: []
PASS: BOOL mean
RESULT: [0.5]
TYPE: %d
SHAPE: []
PASS: Error on empty tensor: shape information must be an array of positive integers
PASS: Error on non-binary bool tensor: validation of data according to the shape provided has failed