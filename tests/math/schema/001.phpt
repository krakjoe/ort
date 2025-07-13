--TEST--
ONNX\Math\Schema: test constructor
--FILE--
<?php
use ONNX\Math;

foreach ([
    "add",
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
    "sqrt",
    "sub",
    "sum",
] as $name)
    new Math\Schema($name);

echo "OK";
?>
--EXPECT--
OK