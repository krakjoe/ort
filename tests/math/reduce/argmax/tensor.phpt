--TEST--
ORT\Math\reduce\tensor\argmax: all types, error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Argmax for full tensor
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\tensor\argmax($a);
        echo "PASS: $name argmax tensor\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name argmax tensor: ".$e->getMessage()."\n";
    }
}

// 2. Error: not a tensor
try {
    $result = ORT\Math\reduce\tensor\argmax([1,2,3]);
    echo "FAIL: Did not throw on non-tensor\n";
} catch (Throwable $e) {
    echo "PASS: Error on non-tensor: ".$e->getMessage()."\n";
}
?>
--EXPECTF--
PASS: FLOAT16 argmax tensor
RESULT: [5]
TYPE: INT64
SHAPE: []
PASS: FLOAT32 argmax tensor
RESULT: [5]
TYPE: INT64
SHAPE: []
PASS: FLOAT64 argmax tensor
RESULT: [5]
TYPE: INT64
SHAPE: []
PASS: INT8 argmax tensor
RESULT: [5]
TYPE: INT64
SHAPE: []
PASS: INT16 argmax tensor
RESULT: [5]
TYPE: INT64
SHAPE: []
PASS: INT32 argmax tensor
RESULT: [5]
TYPE: INT64
SHAPE: []
PASS: INT64 argmax tensor
RESULT: [5]
TYPE: INT64
SHAPE: []
PASS: UINT8 argmax tensor
RESULT: [5]
TYPE: INT64
SHAPE: []
PASS: UINT16 argmax tensor
RESULT: [5]
TYPE: INT64
SHAPE: []
PASS: UINT32 argmax tensor
RESULT: [5]
TYPE: INT64
SHAPE: []
PASS: BOOL argmax tensor
RESULT: [0]
TYPE: INT64
SHAPE: []
PASS: Error on non-tensor: %s
