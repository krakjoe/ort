--TEST--
ONNX\Math\arctanh: all types, shape, error handling, numpy semantics
--FILE--
<?php
use ONNX\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. arctanh for real types: values in (-1, 1)
$values = array_merge(range(-0.99, 0.99, 0.1));
foreach ($real as $name => $type) {
    $a = new ONNX\Tensor\Transient([count($values)], $values, $type);
    $result = ONNX\Math\arctanh($a);
    echo "PASS: $name arctanh real (-1..1)\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT arctanh real (-1..1)
RESULT: %s
TYPE: FLOAT
SHAPE: [%d]
PASS: DOUBLE arctanh real (-1..1)
RESULT: %s
TYPE: DOUBLE
SHAPE: [%d]
