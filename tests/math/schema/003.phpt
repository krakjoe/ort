--TEST--
ORT\Math\Schema: getSymbol
--FILE--
<?php
use ORT\Math;

$schema = new Math\Schema("add");

if ($schema->getSymbol() == "add") {
    echo "OK";
}
?>
--EXPECT--
OK