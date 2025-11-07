--TEST--
Check Tensor with mixed numeric types for DOUBLE
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "mixed_tensor",
    [2, 2],
    [[1, 2.5], [3.7, 4]], // mix of int and float
    ORT\Tensor::FLOAT64
);
echo "OK";
?>
--EXPECT--
OK