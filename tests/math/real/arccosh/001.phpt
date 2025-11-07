--TEST--
ORT\Math\arccosh: all types, shape, error handling, numpy semantics
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. arccosh for real types: values >= 1
$values = array_merge(range(1, 5, 0.1));
foreach ($real as $name => $type) {
    $a = new ORT\Tensor\Transient([count($values)], $values, $type);
    $result = ORT\Math\arccosh($a);
    echo "PASS: $name arccosh real [1..5]\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT16 arccosh real [1..5]
RESULT: %s
TYPE: FLOAT16
SHAPE: [%d]
PASS: FLOAT32 arccosh real [1..5]
RESULT: %s
TYPE: FLOAT32
SHAPE: [%d]
PASS: FLOAT64 arccosh real [1..5]
RESULT: %s
TYPE: FLOAT64
SHAPE: [%d]
