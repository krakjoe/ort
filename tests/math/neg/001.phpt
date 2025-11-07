--TEST--
ORT\Math\neg: tensor negation, all supported types, shape, numpy/onnx semantics, includes large tensor for vectorization
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

$signed_values = array_merge(range(-8, -1), [0], range(1, 8)); // 17 elements
$unsigned_values = range(0, 7); // 8 elements for unsigned
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
    $a = new ORT\Tensor\Transient([count($values)], $values, $type);
    $result = ORT\Math\neg($a);
    echo "PASS: $name neg tensor\n";
    print_result($result);
}

// Large tensor case for vectorization (only supported types)
$large_size = 256;
foreach ($types as $name => [$type, $values]) {
    $a = new ORT\Tensor\Transient([$large_size], array_fill(0, $large_size, 1), $type);
    $result = ORT\Math\neg($a);
    $data = $result->getData();
    echo "PASS: $name neg large tensor (vectorized)\n";
    printf("RESULT: [%d] first=%.1f last=%.1f\n", $large_size, $data[0], $data[$large_size-1]);
    printf("TYPE: %s\n", $name);
    printf("SHAPE: [%d]\n", $large_size);
}
?>
--EXPECTF--
--EXPECTF--
PASS: FLOAT16 neg tensor
RESULT: %s
TYPE: FLOAT16
SHAPE: [17]
PASS: FLOAT32 neg tensor
RESULT: %s
TYPE: FLOAT32
SHAPE: [17]
PASS: FLOAT64 neg tensor
RESULT: %s
TYPE: FLOAT64
SHAPE: [17]
PASS: INT8 neg tensor
RESULT: %s
TYPE: INT8
SHAPE: [17]
PASS: INT16 neg tensor
RESULT: %s
TYPE: INT16
SHAPE: [17]
PASS: INT32 neg tensor
RESULT: %s
TYPE: INT32
SHAPE: [17]
PASS: INT64 neg tensor
RESULT: %s
TYPE: INT64
SHAPE: [17]
PASS: UINT8 neg tensor
RESULT: %s
TYPE: UINT8
SHAPE: [8]
PASS: UINT16 neg tensor
RESULT: %s
TYPE: UINT16
SHAPE: [8]
PASS: UINT32 neg tensor
RESULT: %s
TYPE: UINT32
SHAPE: [8]
PASS: FLOAT16 neg large tensor (vectorized)
RESULT: [%d] first=-1.0 last=-1.0
TYPE: FLOAT16
SHAPE: [256]
PASS: FLOAT32 neg large tensor (vectorized)
RESULT: [%d] first=-1.0 last=-1.0
TYPE: FLOAT32
SHAPE: [256]
PASS: FLOAT64 neg large tensor (vectorized)
RESULT: [%d] first=-1.0 last=-1.0
TYPE: FLOAT64
SHAPE: [256]
PASS: INT8 neg large tensor (vectorized)
RESULT: [%d] first=-1.0 last=-1.0
TYPE: INT8
SHAPE: [256]
PASS: INT16 neg large tensor (vectorized)
RESULT: [%d] first=-1.0 last=-1.0
TYPE: INT16
SHAPE: [256]
PASS: INT32 neg large tensor (vectorized)
RESULT: [%d] first=-1.0 last=-1.0
TYPE: INT32
SHAPE: [256]
PASS: INT64 neg large tensor (vectorized)
RESULT: [%d] first=-1.0 last=-1.0
TYPE: INT64
SHAPE: [256]
PASS: UINT8 neg large tensor (vectorized)
RESULT: [%d] first=255.0 last=255.0
TYPE: UINT8
SHAPE: [256]
PASS: UINT16 neg large tensor (vectorized)
RESULT: [%d] first=65535.0 last=65535.0
TYPE: UINT16
SHAPE: [256]
PASS: UINT32 neg large tensor (vectorized)
RESULT: [%d] first=4294967295.0 last=4294967295.0
TYPE: UINT32
SHAPE: [256]