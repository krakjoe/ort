--TEST--
ORT\Math\Schema: resolve
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Math;
use ORT\Tensor;
use ORT\Status;

$schema = new Math\Schema("add");

if ($schema->resolve(Tensor::FLOAT32, Tensor::FLOAT32) ==
        Tensor::FLOAT32) {
    echo "PASS: binary schema resolved\n";
}

try {
    $schema->resolve(1);
} catch (Status\Schema\InvalidArguments $ex) {
    var_dump($ex->getMessage());
}

try {
    $schema->resolve("1", "1");
} catch (Status\Schema\InvalidArguments $ex) {
    var_dump($ex->getMessage());
}

$schema = new Math\Schema("abs");

if ($schema->resolve(Tensor::FLOAT32) ==
        Tensor::FLOAT32) {
    echo "PASS: unary schema resolved\n";
}

try {
    $schema->resolve(1, 2);
} catch (Status\Schema\InvalidArguments $ex) {
    var_dump($ex->getMessage());
}

try {
    $schema->resolve("1");
} catch (Status\Schema\InvalidArguments $ex) {
    var_dump($ex->getMessage());
}
?>
--EXPECT--
PASS: binary schema resolved
string(60) "binary schemas require two arguments for resolution, 1 given"
string(52) "binary schemas require both arguments to be integers"
PASS: unary schema resolved
string(58) "unary schemas require one argument for resolution, 2 given"
string(41) "unary schemas require an integer argument"