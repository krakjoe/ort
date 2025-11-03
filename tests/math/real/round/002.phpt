--TEST--
ORT\Math\round: large vector, gpu
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
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. Round for real types: negatives, zero, positives
$values = array_merge(range(-16.0, -1.0), [0], range(1.0, 16.0));

$types = [
    'FLOAT' => [$real['FLOAT'], $values],
    'DOUBLE' => [$real['DOUBLE'], $values],
];

// Large tensor case for vectorization
$large_size = 4096;
foreach ($types as $name => [$type, $values]) {
    $a = new ORT\Tensor\Transient([$large_size], array_fill(0, $large_size, 1), $type);
    $result = ORT\Math\round($a);
    $data = $result->getData();
    echo "PASS: $name round large tensor (gpu)\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT round large tensor (gpu)
RESULT: %s
TYPE: FLOAT
SHAPE: [4096]
PASS: DOUBLE round large tensor (gpu)
RESULT: %s
TYPE: DOUBLE
SHAPE: [4096]
