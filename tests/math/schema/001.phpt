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
    "mod",
    "mul",
    "neg",
    "recip",
    "sign",
    "sqrt",
    "sub"
] as $name)
    new Math\Schema($name);

echo "OK";
?>
--EXPECT--
OK