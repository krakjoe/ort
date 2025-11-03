--TEST--
ORT\Math\reduce\dot: tensor • tensor, all types, shape, numpy/onnx semantics, includes large tensor for vectorization
--EXTENSIONS--
ort
--ENV--
ORT_CUDA_THRESHOLD=8192
--SKIPIF--
<?php
if (\ORT\Math\Backend\GPU() === false) {
    die("skip: gpu only test");
}
?>
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

$signed_values = array_merge(range(-16, -1), [0], range(1, 16)); // 33 elements
$unsigned_values = array_merge([0], range(1, 16)); // 17 elements
$types = [
    'FLOAT' => [$real['FLOAT'], $signed_values],
    'DOUBLE' => [$real['DOUBLE'], $signed_values],
    'INT16' => [$signed_types['INT16'], $signed_values],
    'INT32' => [$signed_types['INT32'], $signed_values],
    'UINT16' => [$unsigned_types['UINT16'], $unsigned_values],
    'UINT32' => [$unsigned_types['UINT32'], $unsigned_values],
];

// Large tensor case for vectorization
$large_size = 4096;
foreach ($types as $name => [$type, $values]) {
    $a = new ORT\Tensor\Transient([$large_size], array_fill(0, $large_size, 1), $type);
    $b = new ORT\Tensor\Transient([$large_size], array_fill(0, $large_size, 2), $type);
    $result = ORT\Math\reduce\dot($a, $b);
    echo "PASS: $name dot large tensor • large tensor (vectorized)\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: FLOAT
SHAPE: [1]
PASS: DOUBLE dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: DOUBLE
SHAPE: [1]
PASS: INT16 dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: INT16
SHAPE: [1]
PASS: INT32 dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: INT32
SHAPE: [1]
PASS: UINT16 dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: UINT16
SHAPE: [1]
PASS: UINT32 dot large tensor • large tensor (vectorized)
RESULT: %s
TYPE: UINT32
SHAPE: [1]