--TEST--
Check Tensor with FLOAT64 type
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "double_tensor",
    [1, 3],
    [[1.123456789, 2.987654321, 3.141592653]],
    ORT\Tensor::FLOAT64
);
echo "Type: " . $tensor->getTypeName();
?>
--EXPECT--
Type: FLOAT64