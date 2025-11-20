--TEST--
ORT\Math\pow: tensor ** tensor, large tensor for vectorization (gpu only)
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

$types = [
    'FLOAT16' => [$real['FLOAT16'], [2.0, 1.0]],
    'FLOAT32' => [$real['FLOAT32'], [2.0, 1.0]],
    'FLOAT64' => [$real['FLOAT64'], [2.0, 1.0]],
];

// Large tensor case for vectorization (only supported types)
foreach ($types as $name => [$type, $values]) {
    $large_size = $large_sizes[$name];
    $a = new ORT\Tensor\Transient([$large_size], array_fill(0, $large_size, $values[0]), $type);
    $b = new ORT\Tensor\Transient([$large_size], array_fill(0, $large_size, $values[1]), $type);
    $result = ORT\Math\pow($a, $b);
    $data = $result->getData();
    echo "PASS: $name pow large tensor (gpu)\n";
    printf("RESULT: [%d] first=%.1f last=%.1f\n", $large_size, $data[0], $data[$large_size-1]);
    printf("TYPE: %s\n", $name);
    printf("SHAPE: [%d]\n", $large_size);
}
?>
--EXPECTF--
PASS: FLOAT16 pow large tensor (gpu)
RESULT: [%d] first=2.0 last=2.0
TYPE: FLOAT16
SHAPE: [8192]
PASS: FLOAT32 pow large tensor (gpu)
RESULT: [%d] first=2.0 last=2.0
TYPE: FLOAT32
SHAPE: [4096]
PASS: FLOAT64 pow large tensor (gpu)
RESULT: [%d] first=2.0 last=2.0
TYPE: FLOAT64
SHAPE: [4096]