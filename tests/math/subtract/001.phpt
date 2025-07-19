--TEST--
ORT\Math\subtract: tensor - tensor, all types, shape, numpy/onnx semantics, includes large tensor for vectorization
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
    $a = new ORT\Tensor\Transient([count($values)], $values, $type);
    $b = new ORT\Tensor\Transient([count($values)], $values, $type);
    $result = ORT\Math\subtract($a, $b);
    echo "PASS: $name sub tensor - tensor\n";
    print_result($result);
}

// Large tensor case for vectorization
$large_size = 4096;
foreach ($types as $name => [$type, $values]) {
    $a = new ORT\Tensor\Transient([$large_size], array_fill(0, $large_size, 3), $type);
    $b = new ORT\Tensor\Transient([$large_size], array_fill(0, $large_size, 2), $type);
    $result = ORT\Math\subtract($a, $b);
    $data = $result->getData();

    echo "PASS: $name sub large tensor - large tensor (vectorized)\n";
    printf("RESULT: [%d x %d] first=%.1f last=%.1f\n", $large_size, $large_size, $data[0], $data[$large_size-1]);
    printf("TYPE: %s\n", $name);
    printf("SHAPE: [%d]\n", $large_size);
}
?>
--EXPECTF--
PASS: FLOAT sub tensor - tensor
RESULT: %s
TYPE: FLOAT
SHAPE: [33]
PASS: DOUBLE sub tensor - tensor
RESULT: %s
TYPE: DOUBLE
SHAPE: [33]
PASS: INT8 sub tensor - tensor
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT16 sub tensor - tensor
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT32 sub tensor - tensor
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT64 sub tensor - tensor
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: UINT8 sub tensor - tensor
RESULT: %s
TYPE: UINT8
SHAPE: [17]
PASS: UINT16 sub tensor - tensor
RESULT: %s
TYPE: UINT16
SHAPE: [17]
PASS: UINT32 sub tensor - tensor
RESULT: %s
TYPE: UINT32
SHAPE: [17]
PASS: FLOAT sub large tensor - large tensor (vectorized)
RESULT: [%d x %d] first=1.0 last=1.0
TYPE: FLOAT
SHAPE: [4096]
PASS: DOUBLE sub large tensor - large tensor (vectorized)
RESULT: [%d x %d] first=1.0 last=1.0
TYPE: DOUBLE
SHAPE: [4096]
PASS: INT8 sub large tensor - large tensor (vectorized)
RESULT: [%d x %d] first=1.0 last=1.0
TYPE: INT8
SHAPE: [4096]
PASS: INT16 sub large tensor - large tensor (vectorized)
RESULT: [%d x %d] first=1.0 last=1.0
TYPE: INT16
SHAPE: [4096]
PASS: INT32 sub large tensor - large tensor (vectorized)
RESULT: [%d x %d] first=1.0 last=1.0
TYPE: INT32
SHAPE: [4096]
PASS: INT64 sub large tensor - large tensor (vectorized)
RESULT: [%d x %d] first=1.0 last=1.0
TYPE: INT64
SHAPE: [4096]
PASS: UINT8 sub large tensor - large tensor (vectorized)
RESULT: [%d x %d] first=1.0 last=1.0
TYPE: UINT8
SHAPE: [4096]
PASS: UINT16 sub large tensor - large tensor (vectorized)
RESULT: [%d x %d] first=1.0 last=1.0
TYPE: UINT16
SHAPE: [4096]
PASS: UINT32 sub large tensor - large tensor (vectorized)
RESULT: [%d x %d] first=1.0 last=1.0
TYPE: UINT32
SHAPE: [4096]
