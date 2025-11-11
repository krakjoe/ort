--TEST--
ORT\Math\recip: large vector, gpu
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

// 1. recip for real types: negatives, zero, positives
$values = array_merge(range(-16.0, -1.0), [0], range(1.0, 16.0));

$types = [
    'FLOAT16' => [$real['FLOAT16'], $values],
    'FLOAT32' => [$real['FLOAT32'], $values],
    'FLOAT64' => [$real['FLOAT64'], $values],
];

// Large tensor case for vectorization
foreach ($types as $name => [$type, $values]) {
    $large_size = $large_sizes[$name];
    $a = new ORT\Tensor\Transient([$large_size], array_fill(0, $large_size, 1), $type);
    $result = ORT\Math\recip($a);
    $data = $result->getData();
    echo "PASS: $name recip large tensor (gpu)\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT16 recip large tensor (gpu)
RESULT: %s
TYPE: FLOAT16
SHAPE: [8192]
PASS: FLOAT32 recip large tensor (gpu)
RESULT: %s
TYPE: FLOAT32
SHAPE: [4096]
PASS: FLOAT64 recip large tensor (gpu)
RESULT: %s
TYPE: FLOAT64
SHAPE: [4096]
