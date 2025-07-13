--TEST--
ONNX\Math\Schema: resolve
--FILE--
<?php
use ONNX\Math;
use ONNX\Tensor;

$schema = new Math\Schema("add");

if ($schema->resolve(Tensor::FLOAT, Tensor::FLOAT) ==
        Tensor::FLOAT) {
    echo "OK";
}
?>
--EXPECT--
OK