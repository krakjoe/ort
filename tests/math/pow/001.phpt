--TEST--
ORT\Math\pow: tensor ** tensor, all types, shape, numpy/onnx semantics
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

$values = array_merge(range(-16, -1), [0], range(1, 16));
$scalars = [-7, -3, -1, 0, 1, 3, 7];
foreach (array_merge($real, $signed_types, $unsigned_types) as $name => $type) {
    $a = new ORT\Tensor\Transient([count($values)], $values, $type);
    $b = new ORT\Tensor\Transient([count($values)], $values, $type);
    $result = ORT\Math\pow($a, $b);
    echo "PASS: $name pow tensor ** tensor\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT pow tensor ** tensor
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: DOUBLE pow tensor ** tensor
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT8 pow tensor ** tensor
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT16 pow tensor ** tensor
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT32 pow tensor ** tensor
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT64 pow tensor ** tensor
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: UINT8 pow tensor ** tensor
RESULT: %s
TYPE: UINT8
SHAPE: [33]
PASS: UINT16 pow tensor ** tensor
RESULT: %s
TYPE: UINT16
SHAPE: [33]
PASS: UINT32 pow tensor ** tensor
RESULT: %s
TYPE: UINT32
SHAPE: [33]
