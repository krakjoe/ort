--TEST--
ORT\Math\arctan: all types, shape, error handling, numpy semantics
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. arctan for real types: values in [-5, 5]
$values = array_merge(range(-5, 5, 0.1));
foreach ($real as $name => $type) {
    $a = new ORT\Tensor\Transient([count($values)], $values, $type);
    $result = ORT\Math\arctan($a);
    echo "PASS: $name arctan real [-5..5]\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT16 arctan real [-5..5]
RESULT: %s
TYPE: FLOAT16
SHAPE: [%d]
PASS: FLOAT32 arctan real [-5..5]
RESULT: %s
TYPE: FLOAT32
SHAPE: [%d]
PASS: FLOAT64 arctan real [-5..5]
RESULT: %s
TYPE: FLOAT64
SHAPE: [%d]
