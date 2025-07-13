--TEST--
ONNX\Math\Schema: getSymbol
--FILE--
<?php
use ONNX\Math;

$schema = new Math\Schema("add");

if ($schema->getSymbol() == "add") {
    echo "OK";
}
?>
--EXPECT--
OK