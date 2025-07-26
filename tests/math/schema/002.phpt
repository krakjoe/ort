--TEST--
ORT\Math\Schema: getKind
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Math;

$schema = new Math\Schema("add");

if ($schema->getKind() == Math\Schema::BINARY) {
    echo "OK";
}
?>
--EXPECT--
OK