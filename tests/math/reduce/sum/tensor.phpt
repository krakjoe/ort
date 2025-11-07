--TEST--
ORT\Math\reduce\tensor\sum: all types, shape, and error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

$types = [
    'FLOAT16'  => ORT\Tensor::FLOAT16,
    'FLOAT32'   => ORT\Tensor::FLOAT32,
    'FLOAT64'  => ORT\Tensor::FLOAT64,
    'INT8'    => ORT\Tensor::INT8,
    'INT16'   => ORT\Tensor::INT16,
    'INT32'   => ORT\Tensor::INT32,
    'INT64'   => ORT\Tensor::INT64,
    /* compliance, sum(unsigned) -> uint64
        we don't support uint64, so cannot
        support these operations with numpy semantics
    'UINT8'   => ORT\Tensor::UINT8,
    'UINT16'  => ORT\Tensor::UINT16,
    'UINT32'  => ORT\Tensor::UINT32,
    */
    'BOOL'    => ORT\Tensor::BOOL,
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
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\tensor\sum($a);
        echo "PASS: $name sum\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name sum: ".$e->getMessage()."\n";
    }
}

// 2. Error: empty tensor
try {
    $a = new ORT\Tensor\Transient([0], [], ORT\Tensor::FLOAT32);
    $result = ORT\Math\reduce\tensor\sum($a);
    echo "FAIL: Did not throw on empty tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on empty tensor: ".$e->getMessage()."\n";
}

// 3. Error: bool sum with non-binary values
try {
    $a = new ORT\Tensor\Transient([2,2], [[1,2],[3,4]], ORT\Tensor::BOOL);
    $result = ORT\Math\reduce\tensor\sum($a);
    echo "FAIL: Did not throw on non-binary bool tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on non-binary bool tensor: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT16 sum
RESULT: [21]
TYPE: %d
SHAPE: []
PASS: FLOAT32 sum
RESULT: [21]
TYPE: %d
SHAPE: []
PASS: FLOAT64 sum
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
PASS: Error on non-binary bool tensor: tensor leaf at depth 2: expected bool, got int
