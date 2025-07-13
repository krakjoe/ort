--TEST--
ONNX\Math\Schema: getKind
--FILE--
<?php
use ONNX\Math;

$schema = new Math\Schema("add");

if ($schema->getKind() == Math\Schema::BINARY) {
    echo "OK";
}
?>
--EXPECT--
OK