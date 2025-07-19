--TEST--
ORT\Math\Schema: resolve
--FILE--
<?php
use ORT\Math;
use ORT\Tensor;

$schema = new Math\Schema("add");

if ($schema->resolve(Tensor::FLOAT, Tensor::FLOAT) ==
        Tensor::FLOAT) {
    echo "OK";
}
?>
--EXPECT--
OK