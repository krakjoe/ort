--TEST--
ORT\Math\neg: tensor negation, large tensor for vectorization (gpu only)
--EXTENSIONS--
ort
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
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

$signed_values = array_merge(range(-8, -1), [0], range(1, 8)); // 17 elements
$unsigned_values = range(0, 7); // 8 elements for unsigned
$types = [
    'FLOAT32' => [$real['FLOAT32'], $signed_values],
    'FLOAT64' => [$real['FLOAT64'], $signed_values],
];

// Large tensor case for vectorization (only supported types)
$large_size = 4096;
foreach ($types as $name => [$type, $values]) {
    $a = new ORT\Tensor\Transient([$large_size], array_fill(0, $large_size, 1), $type);
    $result = ORT\Math\neg($a);
    $data = $result->getData();
    echo "PASS: $name neg large tensor (gpu)\n";
    printf("RESULT: [%d] first=%.1f last=%.1f\n", $large_size, $data[0], $data[$large_size-1]);
    printf("TYPE: %s\n", $name);
    printf("SHAPE: [%d]\n", $large_size);
}
?>
--EXPECTF--
PASS: FLOAT32 neg large tensor (gpu)
RESULT: [%d] first=-1.0 last=-1.0
TYPE: FLOAT32
SHAPE: [4096]
PASS: FLOAT64 neg large tensor (gpu)
RESULT: [%d] first=-1.0 last=-1.0
TYPE: FLOAT64
SHAPE: [4096]