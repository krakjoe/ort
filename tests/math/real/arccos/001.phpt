--TEST--
ONNX\Math\arccos: all types, shape, error handling, numpy semantics
--FILE--
<?php
use ONNX\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. arccos for real types: values in [-1, 1]
$values = array_merge(range(-1, 1, 0.1));
foreach ($real as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($values)], $values, $type);
    $result = ONNX\Math\arccos($a);
    echo "PASS: $name arccos real [-1..1]\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT arccos real [-1..1]
RESULT: %s
TYPE: FLOAT
SHAPE: [%d]
PASS: DOUBLE arccos real [-1..1]
RESULT: %s
TYPE: DOUBLE
SHAPE: [%d]
