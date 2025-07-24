--TEST--
ORT\Tensor object handlers
--EXTENSIONS--
ort
--FILE--
<?php
use ORT\Tensor;
use ORT\Status;

$tensor = Tensor\Transient::from([
    [1, 2, 3],
    [4, 5, 6]
], Tensor::INT32);

var_dump(\count($tensor), $tensor[1], $tensor[-1]);

try {
    $tensor[6];
} catch (Status\Tensor\InvalidOffset $e) {
    echo "PASS: rejected invalid offset\n";
}

try {
    $tensor[0] = 42;
} catch (Status\Tensor\InvalidAccess $e) {
    echo "PASS: rejected invalid access\n";
}

try {
    $tensor["0"];
} catch (Status\Tensor\InvalidOffset $e) {
    echo "PASS: rejected invalid offset\n";
}
?>
--EXPECT--
int(6)
int(2)
int(6)
PASS: rejected invalid offset
PASS: rejected invalid access
PASS: rejected invalid offset
