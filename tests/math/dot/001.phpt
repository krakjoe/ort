--TEST--
ONNX\Math\dot: tensor • tensor, all types, shape, numpy/onnx semantics, includes large tensor for vectorization
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
    $a = new ONNX\Tensor\Transient([count($values)], $values, $type);
    $b = new ONNX\Tensor\Transient([count($values)], $values, $type);
    $result = ONNX\Math\dot($a, $b);
    echo "PASS: $name dot tensor • tensor\n";
    print_result($result);
}

// Large tensor case for vectorization
$large_size = 4096;
foreach ($types as $name => [$type, $values]) {
    $a = new ONNX\Tensor\Transient([$large_size], array_fill(0, $large_size, 1), $type);
    $b = new ONNX\Tensor\Transient([$large_size], array_fill(0, $large_size, 2), $type);
    $result = ONNX\Math\dot($a, $b);
    echo "PASS: $name dot large tensor • large tensor (vectorized)\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT dot tensor • tensor
RESULT: %s
TYPE: FLOAT
SHAPE: [1]
PASS: DOUBLE dot tensor • tensor
RESULT: %s
TYPE: DOUBLE
SHAPE: [1]
PASS: INT8 dot tensor • tensor
RESULT: %s
TYPE: INT8
SHAPE: [1]
PASS: INT16 dot tensor • tensor
RESULT: %s
TYPE: INT16
SHAPE: [1]
PASS: INT32 dot tensor • tensor
RESULT: %s
TYPE: INT32
SHAPE: [1]
PASS: INT64 dot tensor • tensor
RESULT: %s
TYPE: INT64
SHAPE: [1]
PASS: UINT8 dot tensor • tensor
RESULT: %s
TYPE: UINT8
SHAPE: [1]
PASS: UINT16 dot tensor • tensor
RESULT: %s
TYPE: UINT16
SHAPE: [1]
PASS: UINT32 dot tensor • tensor
RESULT: %s
TYPE: UINT32
SHAPE: [1]
PASS: FLOAT dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: FLOAT
SHAPE: [1]
PASS: DOUBLE dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: DOUBLE
SHAPE: [1]
PASS: INT8 dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: INT8
SHAPE: [1]
PASS: INT16 dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: INT16
SHAPE: [1]
PASS: INT32 dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: INT32
SHAPE: [1]
PASS: INT64 dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: INT64
SHAPE: [1]
PASS: UINT8 dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: UINT8
SHAPE: [1]
PASS: UINT16 dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: UINT16
SHAPE: [1]
PASS: UINT32 dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: UINT32
SHAPE: [1]