--TEST--
ONNX\Math\arctan: all types, shape, error handling, numpy semantics
--FILE--
<?php
use ONNX\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. arctan for real types: values in [-5, 5]
$values = array_merge(range(-5, 5, 0.1));
foreach ($real as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($values)], $values, $type);
    $result = ONNX\Math\arctan($a);
    echo "PASS: $name arctan real [-5..5]\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT arctan real [-5..5]
RESULT: %s
TYPE: FLOAT
SHAPE: [%d]
PASS: DOUBLE arctan real [-5..5]
RESULT: %s
TYPE: DOUBLE
SHAPE: [%d]
