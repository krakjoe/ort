--TEST--
ORT\Math\cast: basic test
--EXTENSIONS--
ort
--ENV--
ORT_POOL_CORES=12
--FILE--
<?php
use ORT\Tensor;
use ORT\Math;

include sprintf(
    "%s/../../fixtures/math.php",
    dirname(__FILE__));

$tensor = Tensor\Transient::from(
    array_fill(0, 4096, 1.0), Tensor::FLOAT);

$casted = Math\cast(Tensor::INT8, $tensor);
echo "PASS: Casted FLOAT to INT8\n";
print_result($casted);

$tensor = Tensor\Transient::from(
    array_fill(0, (Math\cores() * 256)*2, 1.0), Tensor::FLOAT);

$casted = Math\cast(Tensor::UINT8, $tensor);
echo "PASS: Casted FLOAT to UINT8 (parallel)\n";
print_result($casted);

$casted = Math\cast(Tensor::FLOAT, $tensor);
echo "PASS: Casted FLOAT to FLOAT (deep)\n";
print_result($casted);
?>
--EXPECTF--
PASS: Casted FLOAT to INT8
RESULT: %s
TYPE: INT8
SHAPE: [4096]
PASS: Casted FLOAT to UINT8 (parallel)
RESULT: %s
TYPE: UINT8
SHAPE: [%d]
PASS: Casted FLOAT to FLOAT (deep)
RESULT: %s
TYPE: FLOAT
SHAPE: [%d]