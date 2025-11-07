--TEST--
ORT\Math\arcsin: all types, shape, error handling, numpy semantics
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. arcsin for real types: values in [-1, 1]
$values = array_merge(range(-1, 1, 0.1));
foreach ($real as $name => $type) {
    $a = new ORT\Tensor\Transient([count($values)], $values, $type);
    $result = ORT\Math\arcsin($a);
    echo "PASS: $name arcsin real [-1..1]\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT16 arcsin real [-1..1]
RESULT: %s
TYPE: FLOAT16
SHAPE: [%d]
PASS: FLOAT32 arcsin real [-1..1]
RESULT: %s
TYPE: FLOAT32
SHAPE: [%d]
PASS: FLOAT64 arcsin real [-1..1]
RESULT: %s
TYPE: FLOAT64
SHAPE: [%d]
