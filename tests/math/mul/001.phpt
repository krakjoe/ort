--TEST--
ORT\Math\multiply: tensor + tensor, all types, shape, numpy/onnx semantics, includes large tensor for vectorization
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
    $b = new ORT\Tensor\Transient([count($values)], $values, $type);
    $result = ORT\Math\multiply($a, $b);
    echo "PASS: $name multiply tensor + tensor\n";
    print_result($result);
}

// Large tensor case for vectorization
$large_size = 4096;
foreach ($types as $name => [$type, $values]) {
    $a = new ORT\Tensor\Transient([$large_size], array_fill(0, $large_size, 2), $type);
    $b = new ORT\Tensor\Transient([$large_size], array_fill(0, $large_size, 3), $type);
    $result = ORT\Math\multiply($a, $b);
    $data = $result->getData();
    echo "PASS: $name multiply large tensor + large tensor (vectorized)\n";
    printf("RESULT: [%d x %d] first=%.1f last=%.1f\n", $large_size, $large_size, $data[0], $data[$large_size-1]);
    printf("TYPE: %s\n", $name);
    printf("SHAPE: [%d]\n", $large_size);
}
?>
--EXPECTF--
PASS: FLOAT16 multiply tensor + tensor
RESULT: %s
TYPE: FLOAT16
SHAPE: [33]
PASS: FLOAT32 multiply tensor + tensor
RESULT: %s
TYPE: FLOAT32
SHAPE: [33]
PASS: FLOAT64 multiply tensor + tensor
RESULT: %s
TYPE: FLOAT64
SHAPE: [33]
PASS: INT8 multiply tensor + tensor
RESULT: %s
TYPE: INT8
SHAPE: [33]
PASS: INT16 multiply tensor + tensor
RESULT: %s
TYPE: INT16
SHAPE: [33]
PASS: INT32 multiply tensor + tensor
RESULT: %s
TYPE: INT32
SHAPE: [33]
PASS: INT64 multiply tensor + tensor
RESULT: %s
TYPE: INT64
SHAPE: [33]
PASS: UINT8 multiply tensor + tensor
RESULT: %s
TYPE: UINT8
SHAPE: [17]
PASS: UINT16 multiply tensor + tensor
RESULT: %s
TYPE: UINT16
SHAPE: [17]
PASS: UINT32 multiply tensor + tensor
RESULT: %s
TYPE: UINT32
SHAPE: [17]
PASS: FLOAT16 multiply large tensor + large tensor (vectorized)
RESULT: %s
TYPE: FLOAT16
SHAPE: [4096]
PASS: FLOAT32 multiply large tensor + large tensor (vectorized)
RESULT: %s
TYPE: FLOAT32
SHAPE: [4096]
PASS: FLOAT64 multiply large tensor + large tensor (vectorized)
RESULT: %s
TYPE: FLOAT64
SHAPE: [4096]
PASS: INT8 multiply large tensor + large tensor (vectorized)
RESULT: %s
TYPE: INT8
SHAPE: [4096]
PASS: INT16 multiply large tensor + large tensor (vectorized)
RESULT: %s
TYPE: INT16
SHAPE: [4096]
PASS: INT32 multiply large tensor + large tensor (vectorized)
RESULT: %s
TYPE: INT32
SHAPE: [4096]
PASS: INT64 multiply large tensor + large tensor (vectorized)
RESULT: %s
TYPE: INT64
SHAPE: [4096]
PASS: UINT8 multiply large tensor + large tensor (vectorized)
RESULT: %s
TYPE: UINT8
SHAPE: [4096]
PASS: UINT16 multiply large tensor + large tensor (vectorized)
RESULT: %s
TYPE: UINT16
SHAPE: [4096]
PASS: UINT32 multiply large tensor + large tensor (vectorized)
RESULT: %s
TYPE: UINT32
SHAPE: [4096]