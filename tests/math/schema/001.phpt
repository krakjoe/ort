--TEST--
ONNX\Math\Schema: test constructor
--FILE--
<?php
use ONNX\Math;

foreach ([
    "add",
    "asin",
    "cos",
    "div",
    "dot",
    "matmul",
    "max",
    "mean",
    "min",
    "mod",
    "mul",
    "neg",
    "pow",
    "recip",
    "sign",
    "sin",
    "softmax",
    "sqrt",
    "sub",
    "sum",
    "tan",
] as $name)
    new Math\Schema($name);

echo "OK";
?>
--EXPECT--
OK