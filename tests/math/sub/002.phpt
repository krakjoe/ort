--TEST--
ONNX\Math\subtract: tensor - scalar, all types, numpy/onnx semantics
--EXTENSIONS--
ort
--FILE--
<?php
use ONNX\Tensor;

include sprintf(
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

$signed_values = array_merge(range(-16, -1), [0], range(1, 16)); // 33 elements
$unsigned_values = array_merge([0], range(1, 16)); // 17 elements
$types = [
    'FLOAT' => [$real['FLOAT'], $signed_values, 2.5],
    'DOUBLE' => [$real['DOUBLE'], $signed_values, 2.5],
    'INT8' => [$signed_types['INT8'], $signed_values, 2],
    'INT16' => [$signed_types['INT16'], $signed_values, 2],
    'INT32' => [$signed_types['INT32'], $signed_values, 2],
    'INT64' => [$signed_types['INT64'], $signed_values, 2],
    'UINT8' => [$unsigned_types['UINT8'], $unsigned_values, 2],
    'UINT16' => [$unsigned_types['UINT16'], $unsigned_values, 2],
    'UINT32' => [$unsigned_types['UINT32'], $unsigned_values, 2],
];
foreach ($types as $name => [$type, $values, $scalar]) {
    $a = new ONNX\Tensor\Transient([count($values)], $values, $type);
    $result = ONNX\Math\subtract($a, $scalar);
    echo "PASS: $name sub tensor - scalar\n";
    print_result($result, $name);
}
?>
--EXPECTF--
PASS: FLOAT sub tensor - scalar
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: DOUBLE sub tensor - scalar
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT8 sub tensor - scalar
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT16 sub tensor - scalar
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT32 sub tensor - scalar
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT64 sub tensor - scalar
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: UINT8 sub tensor - scalar
RESULT: %s
TYPE: UINT8
SHAPE: [17]
PASS: UINT16 sub tensor - scalar
RESULT: %s
TYPE: UINT16
SHAPE: [17]
PASS: UINT32 sub tensor - scalar
RESULT: %s
TYPE: UINT32
SHAPE: [17]
