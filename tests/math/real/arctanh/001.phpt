--TEST--
ORT\Math\arctanh: all types, shape, error handling, numpy semantics
--FILE--
<?php
use ORT\Tensor;

include sprintf(
    "%s/../../../fixtures/math.php",
    dirname(__FILE__));

// 1. arctanh for real types: values in (-1, 1)
$values = array_merge(range(-0.99, 0.99, 0.1));
foreach ($real as $name => $type) {
    $a = new ORT\Tensor\Transient([count($values)], $values, $type);
    $result = ORT\Math\arctanh($a);
    echo "PASS: $name arctanh real (-1..1)\n";
    print_result($result);
}
?>
--EXPECTF--
PASS: FLOAT16 arctanh real (-1..1)
RESULT: %s
TYPE: FLOAT16
SHAPE: [%d]
PASS: FLOAT32 arctanh real (-1..1)
RESULT: %s
TYPE: FLOAT32
SHAPE: [%d]
PASS: FLOAT64 arctanh real (-1..1)
RESULT: %s
TYPE: FLOAT64
SHAPE: [%d]
