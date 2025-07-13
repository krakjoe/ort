--TEST--
ONNX\Math\add: tensor + scalar, all types, shape, numpy/onnx semantics
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
$scalars = [-7, -3, -1, 0, 1, 3, 7];
$types = [
    'FLOAT' => [$real['FLOAT'], $signed_values],
    'DOUBLE' => [$real['DOUBLE'], $signed_values],
    'INT8' => [$signed_types['INT8'], $signed_values],
    'INT16' => [$signed_types['INT16'], $signed_values],
    'INT32' => [$signed_types['INT32'], $signed_values],
    'INT64' => [$signed_types['INT64'], $signed_values],
    'UINT8' => [$unsigned_types['UINT8'], $unsigned_values],
    'UINT16' => [$unsigned_types['UINT16'], $unsigned_values],
    'UINT32' => [$unsigned_types['UINT32'], $unsigned_values],
];

foreach ($types as $name => [$type, $values]) {
    foreach ($scalars as $scalar) {
        $a = new ONNX\Tensor\Transient([count($values)], $values, $type);
        $result = ONNX\Math\add($a, $scalar);
        echo "PASS: $name add tensor + scalar ($scalar)\n";
        print_result($result);
    }
}
?>
--EXPECTF--
PASS: FLOAT add tensor + scalar (-7)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: FLOAT add tensor + scalar (-3)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: FLOAT add tensor + scalar (-1)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: FLOAT add tensor + scalar (0)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: FLOAT add tensor + scalar (1)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: FLOAT add tensor + scalar (3)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: FLOAT add tensor + scalar (7)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: DOUBLE add tensor + scalar (-7)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: DOUBLE add tensor + scalar (-3)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: DOUBLE add tensor + scalar (-1)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: DOUBLE add tensor + scalar (0)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: DOUBLE add tensor + scalar (1)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: DOUBLE add tensor + scalar (3)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: DOUBLE add tensor + scalar (7)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT8 add tensor + scalar (-7)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT8 add tensor + scalar (-3)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT8 add tensor + scalar (-1)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT8 add tensor + scalar (0)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT8 add tensor + scalar (1)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT8 add tensor + scalar (3)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT8 add tensor + scalar (7)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT16 add tensor + scalar (-7)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT16 add tensor + scalar (-3)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT16 add tensor + scalar (-1)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT16 add tensor + scalar (0)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT16 add tensor + scalar (1)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT16 add tensor + scalar (3)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT16 add tensor + scalar (7)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT32 add tensor + scalar (-7)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT32 add tensor + scalar (-3)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT32 add tensor + scalar (-1)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT32 add tensor + scalar (0)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT32 add tensor + scalar (1)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT32 add tensor + scalar (3)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT32 add tensor + scalar (7)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT64 add tensor + scalar (-7)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: INT64 add tensor + scalar (-3)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: INT64 add tensor + scalar (-1)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: INT64 add tensor + scalar (0)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: INT64 add tensor + scalar (1)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: INT64 add tensor + scalar (3)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: INT64 add tensor + scalar (7)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: UINT8 add tensor + scalar (-7)
RESULT: %s
TYPE: UINT8
SHAPE: [17]
PASS: UINT8 add tensor + scalar (-3)
RESULT: %s
TYPE: UINT8
SHAPE: [17]
PASS: UINT8 add tensor + scalar (-1)
RESULT: %s
TYPE: UINT8
SHAPE: [17]
PASS: UINT8 add tensor + scalar (0)
RESULT: %s
TYPE: UINT8
SHAPE: [17]
PASS: UINT8 add tensor + scalar (1)
RESULT: %s
TYPE: UINT8
SHAPE: [17]
PASS: UINT8 add tensor + scalar (3)
RESULT: %s
TYPE: UINT8
SHAPE: [17]
PASS: UINT8 add tensor + scalar (7)
RESULT: %s
TYPE: UINT8
SHAPE: [17]
PASS: UINT16 add tensor + scalar (-7)
RESULT: %s
TYPE: UINT16
SHAPE: [17]
PASS: UINT16 add tensor + scalar (-3)
RESULT: %s
TYPE: UINT16
SHAPE: [17]
PASS: UINT16 add tensor + scalar (-1)
RESULT: %s
TYPE: UINT16
SHAPE: [17]
PASS: UINT16 add tensor + scalar (0)
RESULT: %s
TYPE: UINT16
SHAPE: [17]
PASS: UINT16 add tensor + scalar (1)
RESULT: %s
TYPE: UINT16
SHAPE: [17]
PASS: UINT16 add tensor + scalar (3)
RESULT: %s
TYPE: UINT16
SHAPE: [17]
PASS: UINT16 add tensor + scalar (7)
RESULT: %s
TYPE: UINT16
SHAPE: [17]
PASS: UINT32 add tensor + scalar (-7)
RESULT: %s
TYPE: UINT32
SHAPE: [17]
PASS: UINT32 add tensor + scalar (-3)
RESULT: %s
TYPE: UINT32
SHAPE: [17]
PASS: UINT32 add tensor + scalar (-1)
RESULT: %s
TYPE: UINT32
SHAPE: [17]
PASS: UINT32 add tensor + scalar (0)
RESULT: %s
TYPE: UINT32
SHAPE: [17]
PASS: UINT32 add tensor + scalar (1)
RESULT: %s
TYPE: UINT32
SHAPE: [17]
PASS: UINT32 add tensor + scalar (3)
RESULT: %s
TYPE: UINT32
SHAPE: [17]
PASS: UINT32 add tensor + scalar (7)
RESULT: %s
TYPE: UINT32
SHAPE: [17]
