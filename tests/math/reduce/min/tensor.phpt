--TEST--
ORT\Math\reduce\tensor\min: all types, shape, and error handling
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Basic min for all types (2x3)
foreach ($types as $name => $type) {
    if ($type === ORT\Tensor::BOOL) {
        $a = new ORT\Tensor\Transient([2,3], [[true,false,true],[false,true,false]], $type);
    } else {
        $a = new ORT\Tensor\Transient([2,3], [[1,2,3],[4,5,6]], $type);
    }
    try {
        $result = ORT\Math\reduce\tensor\min($a);
        echo "PASS: $name min\n";
        print_result($result);
    } catch (Throwable $e) {
        echo "FAIL: $name min: ".$e->getMessage()."\n";
    }
}
?>
--EXPECTF--
PASS: FLOAT16 min
RESULT: [1]
TYPE: FLOAT16
SHAPE: []
PASS: FLOAT32 min
RESULT: [1]
TYPE: FLOAT32
SHAPE: []
PASS: FLOAT64 min
RESULT: [1]
TYPE: FLOAT64
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
