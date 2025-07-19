--TEST--
Check Tensor with DOUBLE type
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "double_tensor",
    [1, 3],
    [[1.123456789, 2.987654321, 3.141592653]],
    ORT\Tensor::DOUBLE
);
echo "Type: " . $tensor->getTypeName();
?>
--EXPECT--
Type: DOUBLE