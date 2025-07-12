--TEST--
ONNX\Math\mod: tensor % scalar, all types, shape, numpy/onnx semantics
--EXTENSIONS--
ort
--FILE--
<?php
use ONNX\Tensor;

include sprintf(
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

$values = array_merge(range(-16, -1), [0], range(1, 16));
$scalars = [-7, -3, -1, 0, 1, 3, 7];
// tensor % scalar
foreach (array_merge($real, $signed_types, $unsigned_types) as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($values)], $values, $type);
    foreach ($scalars as $scalar) {
        $result = ONNX\Math\mod($a, $scalar);
        echo "PASS: $name mod tensor % scalar ($scalar)\n";
        print_result($result, $name);
    }
}
?>
--EXPECTF--
PASS: FLOAT mod tensor % scalar (-7)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: FLOAT mod tensor % scalar (-3)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: FLOAT mod tensor % scalar (-1)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: FLOAT mod tensor % scalar (0)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: FLOAT mod tensor % scalar (1)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: FLOAT mod tensor % scalar (3)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: FLOAT mod tensor % scalar (7)
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: DOUBLE mod tensor % scalar (-7)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: DOUBLE mod tensor % scalar (-3)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: DOUBLE mod tensor % scalar (-1)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: DOUBLE mod tensor % scalar (0)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: DOUBLE mod tensor % scalar (1)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: DOUBLE mod tensor % scalar (3)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: DOUBLE mod tensor % scalar (7)
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT8 mod tensor % scalar (-7)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT8 mod tensor % scalar (-3)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT8 mod tensor % scalar (-1)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT8 mod tensor % scalar (0)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT8 mod tensor % scalar (1)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT8 mod tensor % scalar (3)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT8 mod tensor % scalar (7)
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT16 mod tensor % scalar (-7)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT16 mod tensor % scalar (-3)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT16 mod tensor % scalar (-1)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT16 mod tensor % scalar (0)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT16 mod tensor % scalar (1)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT16 mod tensor % scalar (3)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT16 mod tensor % scalar (7)
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT32 mod tensor % scalar (-7)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT32 mod tensor % scalar (-3)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT32 mod tensor % scalar (-1)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT32 mod tensor % scalar (0)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT32 mod tensor % scalar (1)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT32 mod tensor % scalar (3)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT32 mod tensor % scalar (7)
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT64 mod tensor % scalar (-7)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: INT64 mod tensor % scalar (-3)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: INT64 mod tensor % scalar (-1)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: INT64 mod tensor % scalar (0)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: INT64 mod tensor % scalar (1)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: INT64 mod tensor % scalar (3)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: INT64 mod tensor % scalar (7)
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: UINT8 mod tensor % scalar (-7)
RESULT: %s
TYPE: UINT8
SHAPE: [33]
PASS: UINT8 mod tensor % scalar (-3)
RESULT: %s
TYPE: UINT8
SHAPE: [33]
PASS: UINT8 mod tensor % scalar (-1)
RESULT: %s
TYPE: UINT8
SHAPE: [33]
PASS: UINT8 mod tensor % scalar (0)
RESULT: %s
TYPE: UINT8
SHAPE: [33]
PASS: UINT8 mod tensor % scalar (1)
RESULT: %s
TYPE: UINT8
SHAPE: [33]
PASS: UINT8 mod tensor % scalar (3)
RESULT: %s
TYPE: UINT8
SHAPE: [33]
PASS: UINT8 mod tensor % scalar (7)
RESULT: %s
TYPE: UINT8
SHAPE: [33]
PASS: UINT16 mod tensor % scalar (-7)
RESULT: %s
TYPE: UINT16
SHAPE: [33]
PASS: UINT16 mod tensor % scalar (-3)
RESULT: %s
TYPE: UINT16
SHAPE: [33]
PASS: UINT16 mod tensor % scalar (-1)
RESULT: %s
TYPE: UINT16
SHAPE: [33]
PASS: UINT16 mod tensor % scalar (0)
RESULT: %s
TYPE: UINT16
SHAPE: [33]
PASS: UINT16 mod tensor % scalar (1)
RESULT: %s
TYPE: UINT16
SHAPE: [33]
PASS: UINT16 mod tensor % scalar (3)
RESULT: %s
TYPE: UINT16
SHAPE: [33]
PASS: UINT16 mod tensor % scalar (7)
RESULT: %s
TYPE: UINT16
SHAPE: [33]
PASS: UINT32 mod tensor % scalar (-7)
RESULT: %s
TYPE: UINT32
SHAPE: [33]
PASS: UINT32 mod tensor % scalar (-3)
RESULT: %s
TYPE: UINT32
SHAPE: [33]
PASS: UINT32 mod tensor % scalar (-1)
RESULT: %s
TYPE: UINT32
SHAPE: [33]
PASS: UINT32 mod tensor % scalar (0)
RESULT: %s
TYPE: UINT32
SHAPE: [33]
PASS: UINT32 mod tensor % scalar (1)
RESULT: %s
TYPE: UINT32
SHAPE: [33]
PASS: UINT32 mod tensor % scalar (3)
RESULT: %s
TYPE: UINT32
SHAPE: [33]
PASS: UINT32 mod tensor % scalar (7)
RESULT: %s
TYPE: UINT32
SHAPE: [33]
