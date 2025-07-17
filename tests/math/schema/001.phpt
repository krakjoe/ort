--TEST--
ONNX\Math\Schema: test constructor
--FILE--
<?php
use ONNX\Math;

foreach ([
    "abs",
    "acos",
    "add",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "asin",
    "atan",
    "cbrt",
    "ceil",
    "cos",
    "cosh",
    "div",
    "dot",
    "exp",
    "exp2",
    "floor",
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
    "round",
    "sign",
    "sin",
    "sinh",
    "softmax",
    "sqrt",
    "sub",
    "sum",
    "tan",
    "tanh",
    "trunc",
] as $name)
    new Math\Schema($name);

echo "OK";
?>
--EXPECT--
OK