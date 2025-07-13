--TEST--
ONNX\Math\Schema: test constructor
--FILE--
<?php
use ONNX\Math;

foreach ([
    "acos",
    "add",
    "asin",
    "atan",
    "cos",
    "cosh",
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
    "sinh",
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