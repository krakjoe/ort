--TEST--
ONNX\Math\Schema: test constructor
--FILE--
<?php
use ONNX\Math;

foreach ([
    "abs",
    "acos",
    "add",
    "asin",
    "atan",
    "cbrt",
    "cos",
    "cosh",
    "div",
    "dot",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
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
    "tanh",
] as $name)
    new Math\Schema($name);

echo "OK";
?>
--EXPECT--
OK