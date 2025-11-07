--TEST--
ORT\Math\divide: tensor + scalar, all types, shape, numpy/onnx semantics
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

$signed_values = array_merge(range(-16, -1), [0], range(1, 16)); // 33 elements
$unsigned_values = array_merge([0], range(1, 16)); // 17 elements
$scalars = [-7, -3, -1, 0, 1, 3, 7];
$types = [
    'FLOAT16' => [$real['FLOAT16'], $signed_values],
    'FLOAT32' => [$real['FLOAT32'], $signed_values],
    'FLOAT64' => [$real['FLOAT64'], $signed_values],
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
        $a = new ORT\Tensor\Transient([count($values)], $values, $type);
        try {
            $result = ORT\Math\divide($a, $scalar);
            echo "PASS: $name divide tensor + scalar ($scalar)\n";
            print_result($result);
        } catch (Throwable $e) {
            echo "FAIL: $name divide tensor + scalar ($scalar): ".$e->getMessage()."\n";
        }
    }
}
?>
--EXPECTF--
PASS: FLOAT16 divide tensor + scalar (-7)
RESULT: %s
TYPE: FLOAT16
SHAPE: [33]
PASS: FLOAT16 divide tensor + scalar (-3)
RESULT: %s
TYPE: FLOAT16
SHAPE: [33]
PASS: FLOAT16 divide tensor + scalar (-1)
RESULT: %s
TYPE: FLOAT16
SHAPE: [33]
PASS: FLOAT16 divide tensor + scalar (0)
RESULT: %s
TYPE: FLOAT16
SHAPE: [33]
PASS: FLOAT16 divide tensor + scalar (1)
RESULT: %s
TYPE: FLOAT16
SHAPE: [33]
PASS: FLOAT16 divide tensor + scalar (3)
RESULT: %s
TYPE: FLOAT16
SHAPE: [33]
PASS: FLOAT16 divide tensor + scalar (7)
RESULT: %s
TYPE: FLOAT16
SHAPE: [33]
PASS: FLOAT32 divide tensor + scalar (-7)
RESULT: %s
TYPE: FLOAT32
SHAPE: [33]
PASS: FLOAT32 divide tensor + scalar (-3)
RESULT: %s
TYPE: FLOAT32
SHAPE: [33]
PASS: FLOAT32 divide tensor + scalar (-1)
RESULT: %s
TYPE: FLOAT32
SHAPE: [33]
PASS: FLOAT32 divide tensor + scalar (0)
RESULT: %s
TYPE: FLOAT32
SHAPE: [33]
PASS: FLOAT32 divide tensor + scalar (1)
RESULT: %s
TYPE: FLOAT32
SHAPE: [33]
PASS: FLOAT32 divide tensor + scalar (3)
RESULT: %s
TYPE: FLOAT32
SHAPE: [33]
PASS: FLOAT32 divide tensor + scalar (7)
RESULT: %s
TYPE: FLOAT32
SHAPE: [33]
PASS: FLOAT64 divide tensor + scalar (-7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: FLOAT64 divide tensor + scalar (-3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: FLOAT64 divide tensor + scalar (-1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: FLOAT64 divide tensor + scalar (0)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: FLOAT64 divide tensor + scalar (1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: FLOAT64 divide tensor + scalar (3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: FLOAT64 divide tensor + scalar (7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT8 divide tensor + scalar (-7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT8 divide tensor + scalar (-3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT8 divide tensor + scalar (-1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT8 divide tensor + scalar (0)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT8 divide tensor + scalar (1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT8 divide tensor + scalar (3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT8 divide tensor + scalar (7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT16 divide tensor + scalar (-7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT16 divide tensor + scalar (-3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT16 divide tensor + scalar (-1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT16 divide tensor + scalar (0)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT16 divide tensor + scalar (1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT16 divide tensor + scalar (3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT16 divide tensor + scalar (7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT32 divide tensor + scalar (-7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT32 divide tensor + scalar (-3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT32 divide tensor + scalar (-1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT32 divide tensor + scalar (0)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT32 divide tensor + scalar (1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT32 divide tensor + scalar (3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT32 divide tensor + scalar (7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT64 divide tensor + scalar (-7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT64 divide tensor + scalar (-3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT64 divide tensor + scalar (-1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT64 divide tensor + scalar (0)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT64 divide tensor + scalar (1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT64 divide tensor + scalar (3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT64 divide tensor + scalar (7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: UINT8 divide tensor + scalar (-7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT8 divide tensor + scalar (-3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT8 divide tensor + scalar (-1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT8 divide tensor + scalar (0)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT8 divide tensor + scalar (1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT8 divide tensor + scalar (3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT8 divide tensor + scalar (7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT16 divide tensor + scalar (-7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT16 divide tensor + scalar (-3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT16 divide tensor + scalar (-1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT16 divide tensor + scalar (0)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT16 divide tensor + scalar (1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT16 divide tensor + scalar (3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT16 divide tensor + scalar (7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT32 divide tensor + scalar (-7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT32 divide tensor + scalar (-3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT32 divide tensor + scalar (-1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT32 divide tensor + scalar (0)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT32 divide tensor + scalar (1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT32 divide tensor + scalar (3)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: UINT32 divide tensor + scalar (7)
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
