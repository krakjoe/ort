--TEST--
Check Tensor with UINT8 type
--EXTENSIONS--
ort
--FILE--
<?php
$tensor = new ORT\Tensor\Persistent(
    "uint8_tensor",
    [2, 2],
    [[255, 0], [128, 64]],
    ORT\Tensor::UINT8
);
echo "Type: " . $tensor->getTypeName();
?>
--EXPECT--
Type: UINT8