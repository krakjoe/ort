--TEST--
ONNX\Math\Schema: test constructor
--FILE--
<?php
use ONNX\Math;

foreach ([
    "add",
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
] as $name)
    new Math\Schema($name);

echo "OK";
?>
--EXPECT--
OK