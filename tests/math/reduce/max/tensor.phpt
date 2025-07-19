--TEST--
ORT\Math\reduce\tensor\max: all types, shape, and error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Basic max for all types (2x3)
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\tensor\max($a);
        echo "PASS: $name max\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name max: ".$e->getMessage()."\n";
    }
}
?>
--EXPECTF--
PASS: FLOAT max
RESULT: [6]
TYPE: FLOAT
SHAPE: []
PASS: DOUBLE max
RESULT: [6]
TYPE: DOUBLE
SHAPE: []
PASS: INT8 max
RESULT: [6]
TYPE: INT8
SHAPE: []
PASS: INT16 max
RESULT: [6]
TYPE: INT16
SHAPE: []
PASS: INT32 max
RESULT: [6]
TYPE: INT32
SHAPE: []
PASS: INT64 max
RESULT: [6]
TYPE: INT64
SHAPE: []
PASS: UINT8 max
RESULT: [6]
TYPE: UINT8
SHAPE: []
PASS: UINT16 max
RESULT: [6]
TYPE: UINT16
SHAPE: []
PASS: UINT32 max
RESULT: [6]
TYPE: UINT32
SHAPE: []
PASS: BOOL max
RESULT: [true]
TYPE: BOOL
SHAPE: []
