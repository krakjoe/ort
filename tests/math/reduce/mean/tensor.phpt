--TEST--
ONNX\Math\reduce\tensor\mean: all types, shape, and error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ONNX\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

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
?>
--EXPECTF--
PASS: FLOAT mean
RESULT: [3.5]
TYPE: FLOAT
SHAPE: []
PASS: DOUBLE mean
RESULT: [3.5]
TYPE: DOUBLE
SHAPE: []
PASS: INT8 mean
RESULT: [3.5]
TYPE: DOUBLE
SHAPE: []
PASS: INT16 mean
RESULT: [3.5]
TYPE: DOUBLE
SHAPE: []
PASS: INT32 mean
RESULT: [3.5]
TYPE: DOUBLE
SHAPE: []
PASS: INT64 mean
RESULT: [3.5]
TYPE: DOUBLE
SHAPE: []
PASS: UINT8 mean
RESULT: [3.5]
TYPE: DOUBLE
SHAPE: []
PASS: UINT16 mean
RESULT: [3.5]
TYPE: DOUBLE
SHAPE: []
PASS: UINT32 mean
RESULT: [3.5]
TYPE: DOUBLE
SHAPE: []
PASS: BOOL mean
RESULT: [0.5]
TYPE: DOUBLE
SHAPE: []