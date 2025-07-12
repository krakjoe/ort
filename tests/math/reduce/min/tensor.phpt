--TEST--
ONNX\Math\reduce\tensor\min: all types, shape, and error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ONNX\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Basic min for all types (2x3)
foreach ($types as $name => $type) {
    if ($type === ONNX\Tensor::BOOL) {
        $a = new ONNX\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ONNX\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ONNX\Math\reduce\tensor\min($a);
        echo "PASS: $name min\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name min: ".$e->getMessage()."\n";
    }
}
?>
--EXPECTF--
PASS: FLOAT min
RESULT: [1]
TYPE: FLOAT
SHAPE: []
PASS: DOUBLE min
RESULT: [1]
TYPE: DOUBLE
SHAPE: []
PASS: INT8 min
RESULT: [1]
TYPE: INT8
SHAPE: []
PASS: INT16 min
RESULT: [1]
TYPE: INT16
SHAPE: []
PASS: INT32 min
RESULT: [1]
TYPE: INT32
SHAPE: []
PASS: INT64 min
RESULT: [1]
TYPE: INT64
SHAPE: []
PASS: UINT8 min
RESULT: [1]
TYPE: UINT8
SHAPE: []
PASS: UINT16 min
RESULT: [1]
TYPE: UINT16
SHAPE: []
PASS: UINT32 min
RESULT: [1]
TYPE: UINT32
SHAPE: []
PASS: BOOL min
RESULT: [false]
TYPE: BOOL
SHAPE: []
